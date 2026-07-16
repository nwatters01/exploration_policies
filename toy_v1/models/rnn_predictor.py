import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim, hidden_sizes, out_dim):
    """MLP with tanh activations between hidden layers and a linear final layer.

    With ``hidden_sizes`` empty this is a single linear map. The caller applies
    the output nonlinearity (a tanh) itself.
    """
    dims = [in_dim, *hidden_sizes, out_dim]
    modules = []
    for k, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        modules.append(nn.Linear(a, b))
        if k < len(dims) - 2:  # activation between hidden layers, not after the final linear
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class PredictiveRNNLayer(nn.Module):
    """One recurrent level that next-step predicts its own input sequence.

    The latent state evolves with a timeconstant ``tau``, and the recurrence is
    conditioned on the action taken at that timestep (a one-hot code over the
    shift actions; do-nothing is all zeros):

        h(t + 1) = (1 - tau) * h(t) + tanh(tau * encoder(h(t) + noise, input(t), action(t)))

    where ``noise`` is zero-mean Gaussian with standard deviation ``noise_std``,
    injected into the recurrence *only during training*. Since ``action(t)``
    modifies ``input(t + 1)``, feeding it here makes the next-step prediction
    action-conditioned. The decoder reads out a
    prediction of the next input from the current latent *and the previous
    timestep's input*, through a scaled tanh that saturates at +/- ``decoder_scale``:

        prediction(t) = decoder_scale * tanh(decoder(h(t + 1), input(t)))   # predicts input(t + 1)

    The initial state h(0) is produced by an MLP over the first two stimulus
    steps and the state initialization of the layer below (see ``init_state``);
    it is not derived from this layer's own input, so ``forward`` takes h(0)
    as an argument.

    ``tau`` is a fixed timeconstant in [0, 1]; **smaller** values give slower,
    more persistent latent dynamics (tau -> 0 means h barely changes).
    """

    def __init__(self, input_dim, hidden_dim, stimulus_dim, below_dim,
                 action_dim=0, initializer_hidden_sizes=(), tau=0.1, noise_std=0.1,
                 decoder_scale=1.0):
        super().__init__()
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stimulus_dim = stimulus_dim
        self.below_dim = below_dim
        self.action_dim = action_dim
        self.tau = float(tau)
        self.noise_std = float(noise_std)
        self.decoder_scale = float(decoder_scale)

        # Initializer MLP: first two stimulus steps (2 * stimulus_dim) plus the
        # layer-below's initial state (below_dim; 0 for the lowest layer).
        self.initializer = _build_mlp(2 * stimulus_dim + below_dim,
                                      initializer_hidden_sizes, hidden_dim)
        # Encoder conditions on the noisy latent, the input, and the action.
        self.encoder = nn.Linear(hidden_dim + input_dim + action_dim, hidden_dim)
        # Decoder conditions on the current latent and the previous input.
        self.decoder = nn.Linear(hidden_dim + input_dim, input_dim)

    def init_state(self, stimulus_first2, below_state):
        """Produce h(0) = tanh(MLP(first two stimulus steps, below layer's h(0))).

        Args:
            stimulus_first2: (batch, 2 * stimulus_dim) flattened first two steps
                of the stimulus.
            below_state: (batch, below_dim) initial state of the layer below, or
                ``None`` for the lowest layer.
        """
        if below_state is None:
            mlp_in = stimulus_first2
        else:
            mlp_in = torch.cat([stimulus_first2, below_state], dim=-1)
        return torch.tanh(self.initializer(mlp_in))

    def _step(self, h, x_t, a_t):
        """Advance the latent state one timestep given state, input, and action."""
        noisy_h = h
        if self.training and self.noise_std > 0:
            noisy_h = h + self.noise_std * torch.randn_like(h)
        parts = [noisy_h, x_t] if a_t is None else [noisy_h, x_t, a_t]
        pre = self.encoder(torch.cat(parts, dim=-1))
        return (1.0 - self.tau) * h + self.tau * torch.tanh(pre)

    def _decode(self, h_next, x_prev):
        """Read out a prediction from the current latent and the previous input."""
        return self.decoder_scale * torch.tanh(self.decoder(torch.cat([h_next, x_prev], dim=-1)))

    def forward(self, x, actions, h0):
        """Run the layer over an input sequence starting from state ``h0``.

        Args:
            x: tensor of shape (batch, num_steps, input_dim).
            actions: tensor (batch, num_steps, action_dim) of one-hot actions, or
                ``None`` if this layer has no action input.
            h0: tensor (batch, hidden_dim), the initial state (from ``init_state``).

        Returns:
            predictions: tensor (batch, num_steps, input_dim). ``predictions[:, t]``
                is the prediction of ``x[:, t + 1]``, made from h(t + 1) and the
                previous input x[:, t].
            hidden: tensor (batch, num_steps + 1, hidden_dim) of latent states
                h(0), h(1), ..., h(num_steps).
        """
        num_steps = x.shape[1]

        h = h0
        hiddens = [h]
        predictions = []
        for t in range(num_steps):
            x_t = x[:, t]
            a_t = None if actions is None else actions[:, t]
            h = self._step(h, x_t, a_t)
            hiddens.append(h)
            predictions.append(self._decode(h, x_t))

        return torch.stack(predictions, dim=1), torch.stack(hiddens, dim=1)

    def next_step_loss(self, x, predictions):
        """L1 next-step prediction loss for this layer over ``x``.

        ``predictions[:, t]`` predicts ``x[:, t + 1]``, so the last prediction
        has no target and is dropped.
        """
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
    layer's latent sequence as its input -- and next-step predicts it. Every
    layer's recurrence is conditioned on the same per-timestep action.

    State initialization is a bottom-up cascade: each layer's initial state is
    ``tanh`` of an MLP that reads the first two steps of the stimulus and the
    initial state of the layer below (the lowest layer sees only the stimulus).
    ``initializer_hidden_sizes`` is the list of hidden sizes shared by these MLPs.

    Every layer runs over a length-T sequence: a higher layer consumes the
    below layer's latent trajectory h(1..T) (excluding the initial state), which
    keeps all layers aligned to the single length-T action sequence. The latent
    sequence handed up to each layer -- and the below-state fed to each
    initializer -- is detached, so every layer's gradients stay **entirely
    separate**: no gradient ever crosses between layers, and each layer is
    trained purely by its own L1 objective.

    ``hidden_dims`` is a sequence whose length sets the number of layers.
    ``action_dim`` is the one-hot action width (0 disables action input).
    ``tau``, ``noise_std`` and ``decoder_scale`` may each be a single value
    (shared across layers) or a per-layer sequence of matching length.
    """

    def __init__(self, input_dim, hidden_dims, action_dim=0, initializer_hidden_sizes=(64,),
                 tau=0.1, noise_std=0.1, decoder_scale=1.0):
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
        self.action_dim = action_dim
        self.initializer_hidden_sizes = list(initializer_hidden_sizes)

        layers = []
        layer_input_dim = input_dim
        below_dim = 0  # the lowest layer has no layer below
        for i in range(num_layers):
            layers.append(PredictiveRNNLayer(
                input_dim=layer_input_dim, hidden_dim=hidden_dims[i],
                stimulus_dim=input_dim, below_dim=below_dim, action_dim=action_dim,
                initializer_hidden_sizes=self.initializer_hidden_sizes,
                tau=taus[i], noise_std=noise_stds[i], decoder_scale=decoder_scales[i]))
            layer_input_dim = hidden_dims[i]  # the next layer consumes this layer's latents
            below_dim = hidden_dims[i]        # ...and its initializer sees this layer's h(0)
        self.layers = nn.ModuleList(layers)

    @property
    def num_layers(self):
        return len(self.layers)

    def initial_states(self, x):
        """Bottom-up initial states h_i(0) for every layer.

        Each is ``tanh(MLP(first two stimulus steps, detached h(0) of the layer
        below))``. The below-state is detached so initializers of different
        layers do not share gradients.
        """
        if x.shape[1] < 2:
            raise ValueError("stimulus must have at least 2 timesteps to initialize state")
        first2 = x[:, :2].reshape(x.shape[0], -1)  # (batch, 2 * input_dim)

        states = []
        below = None
        for layer in self.layers:
            h0 = layer.init_state(first2, below)
            states.append(h0)
            below = h0.detach()  # keep each layer's initializer gradients separate
        return states

    def forward(self, x, actions=None):
        """Run the whole stack over an input sequence.

        Args:
            x: stimulus tensor (batch, T, input_dim).
            actions: one-hot action tensor (batch, T, action_dim), or ``None``.

        Returns a dict of per-layer lists (index i is layer i):
            inputs      : the sequence fed to each layer (the stimulus for layer 0,
                          and the detached latents h(1..T) of the layer below otherwise)
            predictions : each layer's next-step prediction of its own input
            hidden      : each layer's latent states h(0..T)
        Every layer runs over length T, so the single length-T action sequence
        aligns to all of them.
        """
        h0s = self.initial_states(x)

        inputs, predictions, hidden = [], [], []
        layer_input = x
        for layer, h0 in zip(self.layers, h0s):
            pred, h = layer(layer_input, actions, h0)
            inputs.append(layer_input)
            predictions.append(pred)
            hidden.append(h)
            # The next layer consumes h(1..T) (excluding the initial state, so it
            # stays length T), detached so no gradient crosses between layers.
            layer_input = h[:, 1:].detach()
        return {"inputs": inputs, "predictions": predictions, "hidden": hidden}

    def loss_terms(self, x, actions=None):
        """Per-layer L1 next-step losses, as a list (one entry per layer).

        Each layer consumes the detached latents of the layer below, so the
        terms have disjoint gradient paths: summing them and calling
        ``backward`` backpropagates each term into only its own layer.
        """
        out = self.forward(x, actions)
        return [layer.next_step_loss(inp, pred)
                for layer, inp, pred in zip(self.layers, out["inputs"], out["predictions"])]

    def compute_loss(self, x, actions=None):
        """Total L1 loss summed over all layers."""
        return sum(self.loss_terms(x, actions))
