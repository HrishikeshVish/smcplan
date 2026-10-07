"""PyTorch port of AffineInverseAutoregressiveFlow from google-deepmind/
annealed_flow_transport (flows.py: AutoregressiveMLP, InverseAutogressiveFlow,
AffineInverseAutoregressiveFlow). One map shared by all samples; the shift and
scale of each output dimension are functions of the input (autoregressive), so
different particles move differently while the flow stays a single pushforward
with an exact triangular-Jacobian log-det.

Kept faithful to the reference:
  * weights of shape (D, h_prev, D, h), masked by tril(ones(D, D), k) with
    k = 0 on hidden layers and k = -1 (no self links) on the last layer;
  * leaky_relu between layers (jax default slope 0.01);
  * TruncatedNormal(1/sqrt(h_prev * D)) init, zero biases;
  * identity_init -> last-layer weights zero; bias_last -> last layer has bias;
  * shift = raw[:, 0], scale = raw[:, 1] + 1, y = x * scale + shift,
    log|det| = sum log|scale|.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveMLP(nn.Module):
    def __init__(self, dim, hids_per_dim, include_self_links=False, zero_final=True,
                 bias_last=True):
        super().__init__()
        self.dim = dim
        self.hids = list(hids_per_dim)
        self.include_self_links = include_self_links
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.masks = []
        prev = 1
        last = len(self.hids) - 1
        for li, h in enumerate(self.hids):
            w = torch.empty(dim, prev, dim, h)
            if li == last and zero_final:
                nn.init.zeros_(w)
            else:
                std = 1.0 / math.sqrt(prev * dim)
                nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std)
            self.weights.append(nn.Parameter(w))
            if li == last and not bias_last:
                self.biases.append(nn.Parameter(torch.zeros(dim, h), requires_grad=False))
            else:
                self.biases.append(nn.Parameter(torch.zeros(dim, h)))
            k = -1 if (li == last and not include_self_links) else 0
            self.register_buffer(f"mask_{li}", torch.tril(torch.ones(dim, dim), diagonal=k))
            prev = h

    def forward(self, x):                      # x: (B, D) -> (B, D, h_last)
        hid = x.unsqueeze(-1)                  # (B, D, 1)
        last = len(self.hids) - 1
        for li in range(len(self.hids)):
            mask = getattr(self, f"mask_{li}")
            w = mask[:, None, :, None] * self.weights[li]
            hid_new = torch.einsum("ijkl,bij->bkl", w, hid) + self.biases[li]
            hid = F.leaky_relu(hid_new, 0.01) if li != last else hid_new
        return hid


class AffineIAF(nn.Module):
    """AffineInverseAutoregressiveFlow on flattened samples of dimension D."""

    def __init__(self, dim, intermediate_hids_per_dim=30, num_layers=3,
                 identity_init=True, bias_last=True, shift_only=False, shift_scale=1.0,
                 bounded_shift=0.0):
        super().__init__()
        # shift_only: use just the network's shift head, y = x + shift(x). Still a
        # single shared, position-dependent map, but volume preserving, so
        # log|det dT| = 0 exactly and the scale (which broke the DiagonalAffine
        # variants) is gone.
        self.shift_only = shift_only
        # Constant trust-region scale on the shift: T(x) = x + alpha * s(x).
        # alpha is a CONSTANT, so d T/dx = I + alpha * ds/dx stays triangular with
        # unit diagonal and log|det dT| = 0 exactly (an x-dependent normalisation
        # would break that).
        self.shift_scale = shift_scale
        # bounded_shift = delta > 0: T(x) = x + delta * tanh(s(x)). The per-coordinate
        # displacement is capped at delta by construction, so the fit cannot blow the
        # particles up (alpha=8 reached displacement 2.5 and diverged). tanh is applied
        # elementwise to s(x), which is autoregressive, so dT/dx stays triangular with
        # unit diagonal and log|det dT| = 0 exactly.
        self.bounded_shift = bounded_shift
        self.mlp = AutoregressiveMLP(dim, [intermediate_hids_per_dim] * num_layers + [2],
                                     include_self_links=False, zero_final=identity_init,
                                     bias_last=bias_last)

    def forward(self, x):                      # x: (B, D) -> y (B, D), log|det| (B,)
        raw = self.mlp(x)
        shift = raw[..., 0]
        if self.shift_only:
            d = (self.bounded_shift * torch.tanh(shift) if self.bounded_shift > 0
                 else self.shift_scale * shift)
            return x + d, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        scale = raw[..., 1] + 1.0
        return x * scale + shift, torch.log(torch.abs(scale)).sum(-1)


class ResidualFlow(nn.Module):
    """T(x) = x + delta * tanh(g(x)) with g a spectral-normalised MLP.

    Unlike the autoregressive flows, every output coordinate depends on EVERY input
    coordinate, so the map can give different particles genuinely different
    displacements (an autoregressive shift cannot: its last coordinate's shift is a
    bias, identical for all particles -- which is why distillation stalled at ~70%
    error with 32 particles on cube).

    Invertibility: g is spectral-normalised and scaled by `lip` < 1, and
    delta*tanh(.) is 1-Lipschitz in its argument, so the residual branch is a
    contraction and T is invertible (Banach), as in i-ResNet (Behrmann et al. 2019).

    log|det dT| is NOT computed (it needs a Hutchinson/power-series estimate). Every
    use here runs with flow_logdet_weight=0 and cost-to-go-only weights, so the term
    is unused -- but it must be reported as dropped, not as zero.
    """

    def __init__(self, dim, hidden=256, layers=2, delta=0.2, lip=0.9):
        super().__init__()
        from torch.nn.utils.parametrizations import spectral_norm
        mods, d_in = [], dim
        for _ in range(layers):
            mods += [spectral_norm(nn.Linear(d_in, hidden)), nn.ELU()]
            d_in = hidden
        last = nn.Linear(d_in, dim)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # identity at init
        mods += [last]
        self.g = nn.Sequential(*mods)
        self.delta = delta
        self.lip = lip

    def forward(self, x):
        d = self.delta * torch.tanh(self.lip * self.g(x))
        return x + d, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
