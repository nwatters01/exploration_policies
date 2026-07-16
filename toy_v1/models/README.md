# models — stacked predictive-coding RNNs

Two next-step-prediction models, each a stack of recurrent layers where **layer 0
predicts the stimulus and every higher layer predicts (auto-encodes) the latent
trajectory of the layer below it**. Higher layers consume the *detached* latents
of the layer beneath them, so the layers train on **entirely separate gradients**
(each by its own L1 next-step loss).

- [`rnn_predictor.py`](rnn_predictor.py) — `StackedRNNPredictor`: a **fixed**
  scalar timeconstant `tau` per layer.
- [`rnn_predictor_learned_tau.py`](rnn_predictor_learned_tau.py) —
  `LearnedTauStackedRNNPredictor`: a **learned, per-unit, per-timestep** `tau`,
  plus an L1 regularizer on the mean tau (this is the model used by the demos).

Both use the `(1 - tau)` convention, so **`tau → 0` means slow, persistent
dynamics** (`h` barely changes) and `tau → 1` means fast updating.

Notebooks: [`demo_actions.ipynb`](demo_actions.ipynb) trains the learned-tau model
on policy rollouts (with actions); [`demo_no_actions.ipynb`](demo_no_actions.ipynb)
trains it with no actions and evaluates on out-of-distribution test conditions.

---

## The learned-tau model

Each layer, at each timestep, computes a per-unit timeconstant from the same
inputs the encoder sees, and uses it to mix the previous state with a fresh drive:

```
tau(t)   = sigmoid(tau_encoder( h(t) + noise, input(t), action(t) ))    # per unit, in [0, 1]
h(t+1)   = (1 - tau(t)) * h(t) + tau(t) * tanh( encoder( h(t) + noise, input(t), action(t) ) )
pred(t)  = decoder_scale * tanh( decoder( h(t+1), input(t) ) )          # predicts input(t+1)
```

An L1 term `tau_reg_coeff * mean(tau)` (a scalar, or one coefficient per layer) is
added to the loss, so the model **prefers slow dynamics unless faster updating
actually helps prediction**.

![Learned-tau model schematic](images/learned_tau_schematic.png)

The one detail to notice — highlighted in red above — is that the **decoder reads
the previous `input(t)` directly**, alongside the current latent `h(t+1)`. That
"skip" connection is the key to the puzzle below.

### Action-modulated decoder weights

The decoder's weight matrix is **modulated by the action**. Each layer has an
action embedding that produces coefficients over a learned basis of weight
modulations, so the readout applies a *different linear map per action*:

```
W_dec(action) = dec_weight + sum_k  embed(action)_k * dec_weight_mod[k]
```

`none` (all-zeros one-hot) uses the base weights; `left` / `right` add their own
corrections. This matters because a left/right shift is a **permutation of the
input gated by the action** — a multiplicative interaction that a plain
`decoder(h(t+1), input(t))` (which only sees the action *additively*, via the
latent) cannot represent. With the action feeding it additively, the decoder
learns only the always-present drift shift and gets the action-conditioned re-shift
wrong at action steps; modulating the weights lets it pick the right permutation.
Empirically this drops the layer-0 error *at action steps* from ~0.33 to ~0.04
(roughly the non-action level).

---

## How can a layer's prediction be fast while its latent is (almost) frozen?

When you train the learned-tau model (e.g. on drift), the higher layers drive
their tau close to **0** — their latent `h` becomes nearly **static**, holding an
almost constant value across the whole trial. Yet those same layers still produce
**next-step predictions that change every timestep**. How?

Look at the readout: `pred(t) = decoder( h(t+1), input(t) )`. It depends on two
things:

1. **`h(t+1)`** — the recurrent latent. With `tau ≈ 0` this is frozen, so it
   contributes only a slow, near-constant *context*.
2. **`input(t)`** — the layer's previous input, fed to the decoder **directly**
   through the skip connection. For a higher layer this input is the *layer below's
   latent*, which changes quickly every step.

So the prediction inherits its fast, moment-to-moment variation from `input(t)`,
not from the frozen latent. The decoder essentially learns a fast, mostly
**feed-forward** map "given the current input, what's the next input?", while the
slow latent supplies a stable summary/context on the side.

Why does the model *choose* this? The tau regularizer pushes tau toward 0 (slow)
unless fast state updates reduce prediction error. But the skip connection already
lets the decoder predict the next input well from the current input directly, so
the recurrent state doesn't *need* to churn — and the regularizer freezes it. The
skip connection and the regularizer together produce **slow latents with fast
predictions**.

The figure below (layer 1 of a 2-layer model trained on drift, learned mean
`tau ≈ 0.01`) shows exactly this. The latent (row 2) is frozen horizontal
stripes, yet the prediction (row 4) is as lively as the input (row 1):

![Slow latent, fast prediction](images/slow_latent_fast_prediction.png)

Measured on this trial: the layer-1 **latent** changes by only ~0.0006 per step,
while its **input** changes by ~0.04 per step (about 70× faster) — and the
prediction tracks the input, not the latent.

---

## Next steps

Where this model is headed next:

1. **A task with long-range interaction.** The current tasks are essentially
   local (drift moves a bit, noise is per-cell, bifurcate is symmetric). Build a
   task whose next-step structure depends on **distant** parts of the vector — so
   predicting it *requires* integrating information across the whole field, not
   just a local neighbourhood.

2. **Make each layer local.** Restrict each layer's encoder/decoder to a **local
   receptive field** (e.g. a local convolution, as in `toy_v2`) instead of the
   current fully-connected maps. A single local layer then *cannot* see the
   long-range structure — so capturing it must be forced up through the
   **hierarchy**, where stacked local layers grow an effectively larger receptive
   field. This tests whether the predictive-coding stack learns to represent
   long-range dependencies *compositionally*.

3. **Scale up.** With local layers and a long-range task in place, scale the model
   — deeper stacks, wider layers, longer/larger stimuli — and study how much depth
   is needed to span a given interaction range, and how the learned `tau` and the
   action modulation behave at scale.
