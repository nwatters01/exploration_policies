import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveRNNLayer(nn.Module):
    """One recurrent level that next-step predicts its own input sequence.

    The latent state evolves with a timeconstant ``tau``:

        h(t + 1) = tau * h(t) + tanh((1 - tau) * encoder(h(t) + noise, input(t)))
        h(0)     = tanh(initializer(input(0)))

    where ``noise`` is zero-mean Gaussian with standard deviation ``noise_std``,
    injected into the recurrence *only during training*. The decoder reads out a
    prediction of the next input from the current latent *and the previous
    timestep's input*, through a scaled tanh that saturates at +/- ``decoder_scale``:

        prediction(t) = decoder_scale * tanh(decoder(h(t + 1), input(t)))   # predicts input(t + 1)

    ``tau`` is a fixed timeconstant in [0, 1]; larger values give slower,
    more persistent latent dynamics.
    """

    def __init__(self, input_dim, hidden_dim, tau=0.9, noise_std=0.1, decoder_scale=1.0):
        super().__init__()
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = float(tau)
        self.noise_std = float(noise_std)
        self.decoder_scale = float(decoder_scale)

        self.initializer = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Linear(hidden_dim + input_dim, hidden_dim)
        # Decoder conditions on the current latent and the previous input.
        self.decoder = nn.Linear(hidden_dim + input_dim, input_dim)

    def _step(self, h, x_t):
        """Advance the latent state one timestep given current state and input."""
        noisy_h = h
        if self.training and self.noise_std > 0:
            noisy_h = h + self.noise_std * torch.randn_like(h)
        pre = self.encoder(torch.cat([noisy_h, x_t], dim=-1))
        return self.tau * h + torch.tanh((1.0 - self.tau) * pre)

    def _decode(self, h_next, x_prev):
        """Read out a prediction from the current latent and the previous input."""
        return self.decoder_scale * torch.tanh(self.decoder(torch.cat([h_next, x_prev], dim=-1)))

    def forward(self, x):
        """Run the layer over an input sequence.

        Args:
            x: tensor of shape (batch, num_steps, input_dim).

        Returns:
            predictions: tensor (batch, num_steps, input_dim). ``predictions[:, t]``
                is the prediction of ``x[:, t + 1]``, made from h(t + 1) and the
                previous input x[:, t] (the final prediction looks one step past
                the end of the sequence).
            hidden: tensor (batch, num_steps + 1, hidden_dim) of latent states
                h(0), h(1), ..., h(num_steps).
        """
        num_steps = x.shape[1]

        h = torch.tanh(self.initializer(x[:, 0]))
        hiddens = [h]
        predictions = []
        for t in range(num_steps):
            x_t = x[:, t]
            h = self._step(h, x_t)
            hiddens.append(h)
            predictions.append(self._decode(h, x_t))

        return torch.stack(predictions, dim=1), torch.stack(hiddens, dim=1)

    def next_step_loss(self, x, predictions=None):
        """L1 next-step prediction loss for this layer over ``x``.

        ``predictions[:, t]`` predicts ``x[:, t + 1]``, so the last prediction
        has no target and is dropped. Pass ``predictions`` to reuse an existing
        forward pass; otherwise a fresh one is run.
        """
        if predictions is None:
            predictions, _ = self.forward(x)
        return F.l1_loss(predictions[:, :-1], x[:, 1:])


def _broadcast(value, n, name):
    """Turn a scalar into an n-long list, or validate a per-layer sequence."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} has length {len(value)}, expected {n}")
        return list(value)
    return [value] * n


class StackedRNNPredictor(nn.Module):
    """A stack of any number of predictive RNN layers.

    Layer 0 next-step predicts the stimulus. Each higher layer auto-encodes the
    layer below it *in the same way* -- identical recurrent dynamics, with that
    layer's latent sequence as its input -- and next-step predicts it.

    The latent sequence handed up to each layer is detached, so every layer's
    gradients stay **entirely separate**: no gradient ever crosses between
    layers, and each layer is trained purely by its own L1 objective.

    ``hidden_dims`` is a sequence whose length sets the number of layers.
    ``tau``, ``noise_std`` and ``decoder_scale`` may each be a single value
    (shared across layers) or a per-layer sequence of matching length.
    """

    def __init__(self, input_dim, hidden_dims, tau=0.9, noise_std=0.1, decoder_scale=1.0):
        super().__init__()
        hidden_dims = list(hidden_dims)
        num_layers = len(hidden_dims)
        if num_layers == 0:
            raise ValueError("hidden_dims must contain at least one layer")

        taus = _broadcast(tau, num_layers, "tau")
        noise_stds = _broadcast(noise_std, num_layers, "noise_std")
        decoder_scales = _broadcast(decoder_scale, num_layers, "decoder_scale")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        layer_input_dim = input_dim
        for i in range(num_layers):
            layers.append(PredictiveRNNLayer(
                layer_input_dim, hidden_dims[i], taus[i], noise_stds[i], decoder_scales[i]))
            layer_input_dim = hidden_dims[i]  # the next layer consumes this layer's latents
        self.layers = nn.ModuleList(layers)

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, x):
        """Run the whole stack over an input sequence.

        Returns a dict of per-layer lists (index i is layer i):
            inputs      : the sequence fed to each layer (the stimulus for layer 0,
                          and the detached latents of the layer below otherwise)
            predictions : each layer's next-step prediction of its own input
            hidden      : each layer's latent states h(0..T_i)
        Layer i runs over a sequence of length T + i, since each layer consumes
        the (length T + i) latent trajectory of the layer beneath it.
        """
        inputs, predictions, hidden = [], [], []
        layer_input = x
        for layer in self.layers:
            pred, h = layer(layer_input)
            inputs.append(layer_input)
            predictions.append(pred)
            hidden.append(h)
            # Detach so the layer above never sends gradient down into this one.
            layer_input = h.detach()
        return {"inputs": inputs, "predictions": predictions, "hidden": hidden}

    def loss_terms(self, x):
        """Per-layer L1 next-step losses, as a list (one entry per layer).

        Each layer consumes the detached latents of the layer below, so the
        terms have disjoint gradient paths: summing them and calling
        ``backward`` backpropagates each term into only its own layer.
        """
        out = self.forward(x)
        return [layer.next_step_loss(inp, predictions=pred)
                for layer, inp, pred in zip(self.layers, out["inputs"], out["predictions"])]

    def compute_loss(self, x):
        """Total L1 loss summed over all layers."""
        return sum(self.loss_terms(x))
