"""A gradient-free, local-convolutional Hebbian model (pure numpy).

The model has two layers over ``n_input`` spatial positions:

* **Input layer** -- shape ``[n_input, 2]``. Two channels per position: one whose
  rate is the stimulus ``s`` and one whose rate is ``1 - s``.

* **Hidden layer** -- shape ``[n_input, n_channels]``. Driven by a *local
  convolution* from the input and a *local convolutional recurrence* from the
  hidden layer to itself. Convolutions are circular (wrap-around), so the weights
  are translation-invariant kernels of width ``kernel_size``.

Hidden dynamics (elementwise, with timeconstant ``tau``):

    h(t+1) = tau * h(t) + (1 - tau) * phi(W_in * I(t) + W_h * h(t))

where ``*`` denotes the (circular, local) convolution and ``phi`` is a clipped-
linear activation: zero for inputs <= 0, rising with slope ``act_slope``, and flat
at ``act_slope * act_threshold`` for inputs above ``act_threshold``. Each trial the hidden
state is initialised to every unit's running-mean rate estimate (zeros before any
learning has occurred).

Learning is a gradient-free, correlational (Hebbian) rule applied every step:

* Weights are non-negative and initialised uniformly in [0, 1].
* Each weight gets an *additive* delta from the correlation between its pre- and
  post-synaptic units: ``W += lr * corr``. The correlation is the product of the
  two units' rates after subtracting each unit's running-mean rate (an EMA), so
  it estimates a covariance.
* Each pre-synaptic unit's outgoing weights are then clipped non-negative and
  rescaled to sum to a normalization hyperparameter (``norm_in`` for
  input->hidden, ``norm_h`` for the recurrence).
* Finally every weight is pulled a fraction ``weight_decay`` of the way toward
  its neuron's mean outgoing weight (``norm / (K * C_post)``): above-mean weights
  decay, below-mean weights grow. This is sum-preserving.

Homeostasis: every neuron adapts the slope of its (clipped-linear) activation to
drive its running-mean rate toward a target ``target_rate``, so no neuron stays
silent.

Because the weights are convolutional, "each pre-synaptic unit" is a pre-synaptic
*channel*: all positions share one kernel, and the correlational update for a
kernel entry is the average correlation over positions that share it.
"""

import numpy as np


def _piecewise_linear(x, slope, threshold):
    """Clipped-linear activation: 0 for x<=0, slope*x up to the input ``threshold``,
    then flat at slope*threshold above it."""
    return np.clip(slope * x, 0.0, slope * threshold)


class LocalConvHebbianModel:
    def __init__(self, n_input, n_channels, kernel_size=5, tau=0.6,
                 act_slope=1.0, act_threshold=1.0, target_rate=0.2, slope_lr=0.02,
                 norm_in=1.0, norm_h=1.0, lr_in=0.2, lr_h=0.4,
                 weight_decay=0.0, mean_rate=0.02, seed=None):
        """Build the model.

        Args:
            n_input: number of spatial positions (the stimulus length). The input
                layer is [n_input, 2] and the hidden layer is [n_input, n_channels].
            n_channels: number of hidden channels per position.
            kernel_size: spatial width of the local convolution kernels (must be
                odd so the kernel is centered on its target position). Sets how far
                a hidden unit reaches into its neighbourhood, for both the
                input->hidden and recurrent connections.
            tau: hidden-state timeconstant in [0, 1]. Each step mixes the previous
                state with the new drive as tau*h + (1-tau)*phi(...); larger tau
                gives slower, more persistent dynamics.
            act_slope: initial slope of the activation's linear region. The slope
                is per-neuron and adapts during learning (see target_rate), so this
                is only the starting value.
            act_threshold: input value at which the activation saturates. The
                activation phi is piecewise linear: 0 for x <= 0, slope * x for
                0 < x <= act_threshold, and the constant slope * act_threshold for
                x > act_threshold (with each neuron's own adapted slope).
            target_rate: homeostatic target mean firing rate. Each neuron adapts
                its activation slope to drive its running-mean rate toward this,
                which keeps neurons from going silent.
            slope_lr: rate of the homeostatic slope adaptation
                (``slope += slope_lr * (target_rate - mean_rate)`` per step).
            norm_in: target sum of each input channel's outgoing weights. After
                every update the input->hidden kernel is clipped non-negative and
                rescaled so that, for each of the two input channels, the sum of
                all its outgoing weights (over kernel offsets and hidden channels)
                equals this value.
            norm_h: same as norm_in but for the recurrent (hidden->hidden) kernel,
                normalized per pre-synaptic hidden channel. Kept separate so the
                recurrent drive can be scaled independently of the feedforward one.
            lr_in: learning rate of the additive Hebbian update on the input->hidden
                weights (W += lr_in * correlation).
            lr_h: learning rate of the same rule on the recurrent weights.
            weight_decay: fraction, each step, that every weight is pulled toward
                its neuron's mean outgoing weight -- above-mean weights decay,
                below-mean weights grow, both proportional to their distance from
                the mean: ``w -> w + weight_decay * (mean - w)``. The mean is known
                from the normalization (``norm / (K * C_post)``). 0 disables it; 1
                collapses each neuron's weights to the mean.
            mean_rate: EMA rate in [0, 1] for each unit's running-mean firing rate.
                These means center the pre/post rates when forming the correlation,
                so it estimates a covariance; larger values track the mean faster.
            seed: seed for the numpy RNG used to initialise the weights (and any
                other sampling), for reproducibility. None leaves it unseeded.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd so the kernel is centered")
        self.n_input = n_input
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2
        self.tau = tau
        self.act_threshold = act_threshold
        self.target_rate = target_rate
        self.slope_lr = slope_lr
        self.norm_in = norm_in
        self.norm_h = norm_h
        self.lr_in = lr_in
        self.lr_h = lr_h
        self.weight_decay = weight_decay
        self.mean_rate = mean_rate
        self._rng = np.random.default_rng(seed)

        # Per-neuron activation slope, adapted for homeostasis (see _learn).
        self.act_slope = np.full((n_input, n_channels), float(act_slope))

        # Convolution kernels, shape [K, C_pre, C_post], non-negative.
        self.W_in = self._rng.random((kernel_size, 2, n_channels))
        self.W_h = self._rng.random((kernel_size, n_channels, n_channels))
        self._normalize_weights()

        # Running-mean firing-rate estimates (learned statistic, persist across
        # trials). Unset until the first learning step.
        self.mean_I = None
        self.mean_h = None
        self.reset()

    # ------------------------------------------------------------------ state
    def _initial_state(self):
        """The initial hidden state: each unit's running-mean rate (zeros if unset)."""
        if self.mean_h is None:
            return np.zeros((self.n_input, self.n_channels))
        return self.mean_h.copy()

    def reset(self):
        """Reset the hidden state to each unit's mean activity.

        The state is set to the running-mean firing-rate estimate ``mean_h``
        (zeros before any learning). The mean estimates themselves are the learned
        statistic and are *not* cleared here, so they persist across trials.
        """
        self.h = self._initial_state()

    # ------------------------------------------------------------ convolution
    def _conv(self, x, W):
        """Circular local convolution. x: [N, C_pre], W: [K, C_pre, C_post] -> [N, C_post]."""
        out = np.zeros((x.shape[0], W.shape[2]))
        for k in range(self.kernel_size):
            shift = k - self.center
            # np.roll(x, -shift)[p] == x[(p + shift) % N]
            out += np.roll(x, -shift, axis=0) @ W[k]
        return out

    def _corr_kernel(self, pre_c, post_c):
        """Per-kernel-entry correlation. pre_c: [N, C_pre], post_c: [N, C_post] -> [K, C_pre, C_post].

        For kernel offset ``shift = k - center`` the pre-synaptic unit feeding
        post position ``p`` sits at ``p + shift``; the entry's correlation is the
        average over positions of pre[p+shift] * post[p].
        """
        n = pre_c.shape[0]
        corr = np.empty((self.kernel_size, pre_c.shape[1], post_c.shape[1]))
        for k in range(self.kernel_size):
            shift = k - self.center
            rolled = np.roll(pre_c, -shift, axis=0)
            corr[k] = (rolled.T @ post_c) / n
        return corr

    def _pull_to_mean(self, W, norm):
        """Push every outgoing weight toward its neuron's mean output weight.

        Each weight moves a fraction ``weight_decay`` of the way to the mean, in
        proportion to its distance from it: ``W -> W + weight_decay * (mean - W)``.
        So above-mean weights decay and below-mean weights grow. The per-neuron
        mean output weight is known from the normalization -- each pre-synaptic
        channel's ``K * C_post`` outgoing weights sum to ``norm``, giving a mean of
        ``norm / (K * C_post)``. Applied to already-normalized weights this is
        sum-preserving (and keeps weights non-negative).
        """
        mean_out = norm / (W.shape[0] * W.shape[2])
        return W + self.weight_decay * (mean_out - W)

    def _normalize_weights(self):
        """Clip to non-negative and rescale each pre-synaptic channel's outgoing sum."""
        self.W_in = np.clip(self.W_in, 0.0, None)
        s_in = self.W_in.sum(axis=(0, 2), keepdims=True)          # per input channel
        self.W_in *= self.norm_in / np.where(s_in > 0, s_in, 1.0)

        self.W_h = np.clip(self.W_h, 0.0, None)
        s_h = self.W_h.sum(axis=(0, 2), keepdims=True)            # per hidden (pre) channel
        self.W_h *= self.norm_h / np.where(s_h > 0, s_h, 1.0)

    # ------------------------------------------------------------------ steps
    def step(self, stimulus, learn=True):
        """Advance one timestep. Returns (I, h) with shapes [N, 2] and [N, C]."""
        s = np.asarray(stimulus, dtype=float)
        I = np.stack([s, 1.0 - s], axis=1)

        pre = self._conv(I, self.W_in) + self._conv(self.h, self.W_h)
        # Per-neuron slope (self.act_slope is a [N, C] array).
        activation = _piecewise_linear(pre, self.act_slope, self.act_threshold)
        h_next = self.tau * self.h + (1.0 - self.tau) * activation

        if learn:
            self._learn(I, self.h, h_next)

        self.h = h_next
        return I, h_next

    def _learn(self, I, h_prev, h_next):
        # Update running-mean rate estimates (EMA), lazily initialised.
        if self.mean_I is None:
            self.mean_I = I.copy()
            self.mean_h = h_next.copy()
        else:
            self.mean_I += self.mean_rate * (I - self.mean_I)
            self.mean_h += self.mean_rate * (h_next - self.mean_h)

        # Homeostatic intrinsic plasticity: adapt each neuron's activation slope to
        # drive its mean rate toward the target (raise the gain of quiet neurons,
        # lower it for over-active ones). Non-negative weights make the
        # pre-activation >= 0, so a higher slope reliably raises the rate.
        self.act_slope = np.maximum(
            self.act_slope + self.slope_lr * (self.target_rate - self.mean_h), 0.0)

        I_c = I - self.mean_I
        h_prev_c = h_prev - self.mean_h
        h_next_c = h_next - self.mean_h

        # Additive Hebbian update, renormalize each pre-synaptic channel's outgoing
        # sum, then pull every weight toward its neuron's mean (sum-preserving).
        self.W_in += self.lr_in * self._corr_kernel(I_c, h_next_c)
        self.W_h += self.lr_h * self._corr_kernel(h_prev_c, h_next_c)
        self._normalize_weights()
        if self.weight_decay > 0.0:
            self.W_in = self._pull_to_mean(self.W_in, self.norm_in)
            self.W_h = self._pull_to_mean(self.W_h, self.norm_h)

    def run(self, stimuli, learn=True, reset_hidden=True):
        """Run over a stimulus sequence. stimuli: [T, N] -> (I_seq [T,N,2], h_seq [T,N,C]).

        By default each run is a fresh trial: the hidden state is reset to the
        running-mean activity before the sequence. The running-mean estimates and
        weights are preserved, so learning accumulates across runs.
        """
        if reset_hidden:
            self.reset()
        I_seq, h_seq = [], []
        for s in stimuli:
            I, h = self.step(s, learn=learn)
            I_seq.append(I)
            h_seq.append(h)
        return np.stack(I_seq), np.stack(h_seq)
