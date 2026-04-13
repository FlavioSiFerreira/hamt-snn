# Cross-Sample Habituation: Design Notes

## The Problem

Current HAMT tracks familiarity within a single forward pass (5 timesteps
of history within 25 simulation steps). This means habituation resets for
every new input sample. The network never "remembers" that it has seen
digit 3 ten thousand times before.

Biological habituation operates across exposures over hours, days, and
weeks. A neuron that fires strongly for a novel face on day 1 will fire
weakly for the same face on day 30. Our implementation only habituates
within microseconds of a single inference pass.

Result: HAMT acts as a learned within-sample regularizer (~30% energy
reduction) rather than a true cross-sample habituation mechanism that
would compound savings over training duration.

## Evidence

100-epoch Fashion-MNIST experiment (2026-04-11) shows:
- Energy reduction stabilizes at ~30-35% across all epochs
- Both baseline and HAMT spike rates climb in parallel
- The gap is proportional, not growing
- Habituation is not "learning" to suppress familiar patterns over time

## Proposed Fix: Persistent Familiarity Memory

### Option A: Exponential Moving Average (simplest)

Add a persistent buffer per neuron that tracks the running average of
input patterns across all training samples:

```python
class PersistentHabituationLayer(nn.Module):
    def __init__(self, num_neurons, memory_decay=0.999):
        super().__init__()
        self.memory_decay = memory_decay
        # Persistent across batches (not a parameter, not reset)
        self.register_buffer(
            'input_memory', torch.zeros(num_neurons)
        )
        self.register_buffer(
            'exposure_count', torch.zeros(num_neurons)
        )

    def forward(self, current_input, state):
        # Update persistent memory with EMA
        with torch.no_grad():
            batch_mean = current_input.mean(dim=0)
            self.input_memory = (
                self.memory_decay * self.input_memory
                + (1 - self.memory_decay) * batch_mean
            )
            self.exposure_count += 1

        # Compute novelty against PERSISTENT memory, not just
        # the last 5 timesteps
        long_term_familiarity = torch.abs(
            current_input - self.input_memory
        ) / (torch.abs(self.input_memory) + 1e-8)

        # Combine with short-term (within-sample) familiarity
        # ... rest of habituation logic
```

Advantage: Simple, no extra parameters, memory persists across batches.
Risk: The EMA might converge to the dataset mean, making everything
"familiar" and suppressing too aggressively.

### Option B: Per-Class Prototype Memory

Track a prototype (centroid) for each class the network has seen,
and suppress more aggressively when the current input is close to a
known prototype:

```python
class PrototypeHabituation(nn.Module):
    def __init__(self, num_neurons, num_classes, decay=0.99):
        super().__init__()
        self.register_buffer(
            'prototypes', torch.zeros(num_classes, num_neurons)
        )
        self.register_buffer(
            'prototype_counts', torch.zeros(num_classes)
        )

    def update_prototypes(self, hidden_activations, labels):
        """Call after forward pass with known labels."""
        with torch.no_grad():
            for c in labels.unique():
                mask = labels == c
                class_mean = hidden_activations[mask].mean(dim=0)
                n = self.prototype_counts[c]
                self.prototypes[c] = (
                    n / (n + 1) * self.prototypes[c]
                    + 1 / (n + 1) * class_mean
                )
                self.prototype_counts[c] += mask.sum()
```

Advantage: Novelty is class-aware. A new class gets full processing,
a class seen 10,000 times gets heavy suppression.
Risk: Requires label access during forward pass (only for training).
More complex integration.

### Option C: Hebbian Familiarity Trace

Track which neuron activation patterns co-occur frequently using a
Hebbian-like outer product trace:

```python
# After each batch, accumulate co-activation patterns
self.coactivation = decay * self.coactivation + (1-decay) * (h.T @ h) / batch_size
# High coactivation = familiar pattern = suppress
```

Advantage: Unsupervised, captures structural patterns.
Risk: O(n^2) memory for n neurons, may be too expensive for h=800.

## Recommendation

Start with Option A (EMA). It is the simplest change (add ~10 lines to
HabituationLayer), requires no architectural changes, and directly tests
whether persistent memory improves the compounding effect.

If EMA works (energy reduction grows past 30% at epoch 50+), then
Option B could refine it further with class-aware suppression.

## Impact on Patent

The current patent claims cover "a habituation module that tracks input
familiarity." Adding persistent cross-sample memory strengthens claim 3
(familiarity trace) and could add a new dependent claim about
"maintaining familiarity state across training iterations."

File the provisional patent BEFORE implementing this change, so the
original mechanism is protected. The improvement can be added in the
full utility patent filing.
