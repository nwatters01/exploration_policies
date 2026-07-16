"""A predictive RNN whose timeconstant ``tau`` is learned and dynamic.

Same architecture as ``rnn_predictor.StackedRNNPredictor``, but instead of a
fixed scalar ``tau`` each layer computes a **per-unit tau at every timestep** from
the same inputs the encoder sees, squashed to [0, 1] with a sigmoid:

    tau(t)   = sigmoid(tau_encoder(h(t) + noise, input(t), action(t)))   # shape [batch, hidden_dim]
    h(t + 1) = (1 - tau(t)) * h(t) + tanh(tau(t) * encoder(h(t) + noise, input(t), action(t)))

With the (1 - tau) convention, ``tau -> 0`` means slow, persistent dynamics.

An L1 regularizer on the mean tau (weighted by ``tau_reg_coeff``, optionally
per-layer) is added to the loss, so the model prefers slow dynamics unless faster
updating helps prediction.

Each layer's **decoder weights are modulated by the action**: a per-layer action
embedding produces coefficients over a learned basis of weight modulations, so the
readout applies a different linear map per action. This lets the model predict
action-dependent shifts of its input (which an additive action term cannot do,
since a shift is a permutation gated by the action).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn_predictor import _broadcast, _build_mlp


class LearnedTauPredictiveRNNLayer(nn.Module):
    """One predictive layer with a learned, per-unit, per-timestep ``tau``."""

    def __init__(self, input_dim, hidden_dim, stimulus_dim, below_dim,
                 action_dim=0, action_embed_dim=8, initializer_hidden_sizes=(),
                 noise_std=0.1, decoder_scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stimulus_dim = stimulus_dim
        self.below_dim = below_dim
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim
        self.noise_std = float(noise_std)
        self.decoder_scale = float(decoder_scale)

        self.initializer = _build_mlp(2 * stimulus_dim + below_dim,
                                      initializer_hidden_sizes, hidden_dim)
        # Encoder produces the drive; tau_encoder produces the per-unit timeconstant.
        self.encoder = nn.Linear(hidden_dim + input_dim + action_dim, hidden_dim)
        self.tau_encoder = nn.Linear(hidden_dim + input_dim + action_dim, hidden_dim)

        # Decoder reads out from the current latent and the previous input. Its
        # weight matrix is *modulated by the action*: a per-layer action embedding
        # gives coefficients over a learned basis of weight modulations, so
        # W_dec(action) = dec_weight + sum_k embed(action)_k * dec_weight_mod[k].
        # `none` (all-zeros one-hot) -> base weights; left/right add corrections.
        dec_in = hidden_dim + input_dim
        self.dec_weight = nn.Parameter(torch.empty(input_dim, dec_in))
        nn.init.kaiming_uniform_(self.dec_weight, a=math.sqrt(5))
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))
        if action_dim > 0 and action_embed_dim > 0:
            self.action_embed = nn.Linear(action_dim, action_embed_dim, bias=False)
            self.dec_weight_mod = nn.Parameter(torch.zeros(action_embed_dim, input_dim, dec_in))
        else:
            self.action_embed = None
            self.dec_weight_mod = None

    def init_state(self, stimulus_first2, below_state):
        if below_state is None:
            mlp_in = stimulus_first2
        else:
            mlp_in = torch.cat([stimulus_first2, below_state], dim=-1)
        return torch.tanh(self.initializer(mlp_in))

    def _step(self, h, x_t, a_t):
        """Advance one step, returning (h_next, tau) with tau shape (batch, hidden_dim)."""
        noisy_h = h
        if self.training and self.noise_std > 0:
            noisy_h = h + self.noise_std * torch.randn_like(h)
        parts = [noisy_h, x_t] if a_t is None else [noisy_h, x_t, a_t]
        feat = torch.cat(parts, dim=-1)
        pre = self.encoder(feat)
        tau = torch.sigmoid(self.tau_encoder(feat))
        h_next = (1.0 - tau) * h + tau * torch.tanh(pre)
        return h_next, tau

    def _decode(self, h_next, x_prev, a_t):
        """Decode a prediction; the decoder weights are modulated by the action ``a_t``."""
        feat = torch.cat([h_next, x_prev], dim=-1)                       # (B, dec_in)
        if self.action_embed is not None and a_t is not None:
            embed = self.action_embed(a_t)                              # (B, action_embed_dim)
            weight = self.dec_weight + torch.einsum(
                'bk,koi->boi', embed, self.dec_weight_mod)              # (B, out, in)
            out = torch.einsum('boi,bi->bo', weight, feat) + self.dec_bias
        else:
            out = feat @ self.dec_weight.t() + self.dec_bias
        return self.decoder_scale * torch.tanh(out)

    def forward(self, x, actions, h0):
        """Run the layer. Returns (predictions, hidden, taus).

        predictions: (batch, T, input_dim); hidden: (batch, T + 1, hidden_dim);
        taus: (batch, T, hidden_dim) -- the per-unit tau used at each step.
        """
        num_steps = x.shape[1]
        h = h0
        hiddens = [h]
        predictions, taus = [], []
        for t in range(num_steps):
            x_t = x[:, t]
            a_t = None if actions is None else actions[:, t]
            h, tau = self._step(h, x_t, a_t)
            hiddens.append(h)
            taus.append(tau)
            predictions.append(self._decode(h, x_t, a_t))
        return (torch.stack(predictions, dim=1),
                torch.stack(hiddens, dim=1),
                torch.stack(taus, dim=1))

    def next_step_loss(self, x, predictions):
        return F.l1_loss(predictions[:, :-1], x[:, 1:])


class LearnedTauStackedRNNPredictor(nn.Module):
    """Stack of ``LearnedTauPredictiveRNNLayer``s with an L1 mean-tau regularizer.

    Identical stacking/gradient-separation to ``StackedRNNPredictor`` (higher
    layers consume the detached latents h(1..T) of the layer below). Each layer
    has a learned per-unit tau; the total loss is the sum of the per-layer L1
    next-step losses plus ``tau_reg_coeff[i] * mean(tau_i)`` for each layer.

    ``tau_reg_coeff`` may be a single value (shared) or a per-layer list.
    """

    def __init__(self, input_dim, hidden_dims, action_dim=0, action_embed_dim=8,
                 initializer_hidden_sizes=(64,), noise_std=0.1, decoder_scale=1.0,
                 tau_reg_coeff=0.0):
        super().__init__()
        hidden_dims = list(hidden_dims)
        num_layers = len(hidden_dims)
        if num_layers == 0:
            raise ValueError("hidden_dims must contain at least one layer")

        noise_stds = _broadcast(noise_std, num_layers, "noise_std")
        decoder_scales = _broadcast(decoder_scale, num_layers, "decoder_scale")
        self.tau_reg_coeffs = _broadcast(tau_reg_coeff, num_layers, "tau_reg_coeff")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.initializer_hidden_sizes = list(initializer_hidden_sizes)

        layers = []
        layer_input_dim = input_dim
        below_dim = 0
        for i in range(num_layers):
            layers.append(LearnedTauPredictiveRNNLayer(
                input_dim=layer_input_dim, hidden_dim=hidden_dims[i],
                stimulus_dim=input_dim, below_dim=below_dim, action_dim=action_dim,
                action_embed_dim=action_embed_dim,
                initializer_hidden_sizes=self.initializer_hidden_sizes,
                noise_std=noise_stds[i], decoder_scale=decoder_scales[i]))
            layer_input_dim = hidden_dims[i]
            below_dim = hidden_dims[i]
        self.layers = nn.ModuleList(layers)

    @property
    def num_layers(self):
        return len(self.layers)

    def initial_states(self, x):
        if x.shape[1] < 2:
            raise ValueError("stimulus must have at least 2 timesteps to initialize state")
        first2 = x[:, :2].reshape(x.shape[0], -1)
        states, below = [], None
        for layer in self.layers:
            h0 = layer.init_state(first2, below)
            states.append(h0)
            below = h0.detach()
        return states

    def forward(self, x, actions=None):
        """Run the stack. Returns a dict of per-layer lists:
        inputs, predictions, hidden, taus (per-unit tau over time for each layer)."""
        h0s = self.initial_states(x)
        inputs, predictions, hidden, taus = [], [], [], []
        layer_input = x
        for layer, h0 in zip(self.layers, h0s):
            pred, h, tau = layer(layer_input, actions, h0)
            inputs.append(layer_input)
            predictions.append(pred)
            hidden.append(h)
            taus.append(tau)
            layer_input = h[:, 1:].detach()
        return {"inputs": inputs, "predictions": predictions, "hidden": hidden, "taus": taus}

    def loss_terms(self, x, actions=None):
        """Per-layer L1 next-step losses (list), matching StackedRNNPredictor."""
        out = self.forward(x, actions)
        return [layer.next_step_loss(inp, pred)
                for layer, inp, pred in zip(self.layers, out["inputs"], out["predictions"])]

    def compute_losses(self, x, actions=None):
        """Single forward pass returning everything needed to train and visualize.

        Returns a dict with the ``forward`` outputs plus:
            l1       : list of per-layer L1 next-step losses
            tau_reg  : list of per-layer ``coeff * mean(tau)`` regularizers
            total    : scalar total loss (sum of l1 + sum of tau_reg)
        """
        out = self.forward(x, actions)
        l1 = [layer.next_step_loss(inp, pred)
              for layer, inp, pred in zip(self.layers, out["inputs"], out["predictions"])]
        tau_reg = [coeff * taus.mean() for coeff, taus in zip(self.tau_reg_coeffs, out["taus"])]
        out["l1"] = l1
        out["tau_reg"] = tau_reg
        out["total"] = sum(l1) + sum(tau_reg)
        return out

    def compute_loss(self, x, actions=None):
        """Total loss: per-layer L1 next-step + per-layer mean-tau L1 regularizer."""
        return self.compute_losses(x, actions)["total"]
