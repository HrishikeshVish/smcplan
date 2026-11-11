"""Batched torch ports of blackjax's MCMC kernels, run as P independent chains.

Why a port and not the library: blackjax (installed here, v1.3) is JAX-native,
and the target density is a torch world-model rollout on the GPU. Bridging the
two per kernel step needs `jax.pure_callback` plus a hand-written `custom_vjp`
for the gradient, a host synchronisation every step, and two frameworks
contending for the same card. The kernels are ~15 lines each, so the port is
the cheaper and far more inspectable option -- but a port is worth nothing
unless it is faithful, so every acceptance ratio in this file is checked
against the blackjax function it came from in `test_mcmc_kernels.py`: exact
agreement on constructed state pairs, plus invariance checks on targets whose
moments are known in closed form.

Correspondence with blackjax 1.3:

  ULA   blackjax.mcmc.diffusions.overdamped_langevin   (no accept step)
  MALA  blackjax.mcmc.mala.build_kernel
  RWM   blackjax.mcmc.random_walk.build_rmh            (Gaussian, symmetric)
  HMC   blackjax.mcmc.hmc.build_kernel with
        blackjax.mcmc.integrators.velocity_verlet, identity mass matrix
  step size
        blackjax.adaptation.step_size.dual_averaging_adaptation
        (Hoffman & Gelman 2014, sec 3.2.1; t0=10, gamma=0.05, kappa=0.75)

blackjax's `step_size` convention is kept throughout: the ULA/MALA move is
`x + eps*grad + sqrt(2*eps)*xi`, i.e. eps multiplies the gradient directly
instead of appearing as eps^2/2. fk_planner's `langevin_lr` is the same
quantity, so step sizes transfer between the two files unchanged.

Batching: blackjax advances one chain per kernel call and vmaps for more.
Here the P particles ARE the chains and they advance in lockstep, because the
point of the design is that one kernel step costs exactly one batched
world-model rollout of P sequences. Accept/reject is therefore per-chain -- a
(P,) boolean mask -- while the step size is shared and adapted from the mean
acceptance rate, the usual convention for parallel chains.

What these kernels deliberately do NOT do is optimise. They are pi-invariant
samplers and nothing else: no preconditioner, no momentum, no annealed
objective, no loss smoothing inside the accept step. That is the whole point
of the separation (see em_smc.py) -- fk_planner's kernels failed as samplers
because they were optimisers wearing a sampler's name, and they failed as
optimisers when the optimiser was taken away. Descent lives in the M-step.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch


# ---------------------------------------------------------------- chain state


@dataclass
class ChainState:
    """A chain's position and everything cached at it.

    `cost`/`cost_grad` (and the diagnostic `task`/`value`) ride alongside
    logdensity/grad because the planner moves the target between kernel steps:
    the M-step moves mu and the tempering moves beta, and both act on log pi
    through pure algebra on C (see `ActionSequenceTarget.recompose`). Threading
    them through the accept mask is what makes those two updates cost zero
    world-model rollouts.

    Shapes: position/grad/cost_grad are (P, H, A); the rest are (P,).
    """

    position: torch.Tensor
    logdensity: torch.Tensor
    grad: torch.Tensor
    cost: torch.Tensor
    cost_grad: torch.Tensor
    task: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None

    _FIELDS = ("position", "logdensity", "grad", "cost", "cost_grad",
               "task", "value")

    def index_select(self, idx):
        """Reorder chains (used by resampling). Keeps every cached quantity."""
        def pick(v):
            return None if v is None else v[idx]
        return ChainState(*[pick(getattr(self, f)) for f in self._FIELDS])

    def where(self, accepted, other):
        """Per-chain choice: `self` where `accepted`, `other` elsewhere."""
        def pick(new, old):
            if new is None or old is None:
                return None
            m = accepted.view(-1, *([1] * (new.dim() - 1)))
            return torch.where(m, new, old)
        return ChainState(*[pick(getattr(self, f), getattr(other, f))
                            for f in self._FIELDS])


# ------------------------------------------------------- blackjax primitives


def safe_energy_diff(initial_energy, new_energy):
    """blackjax.mcmc.proposal.safe_energy_diff.

    A NaN energy difference (a diverged rollout, an inf cost) becomes -inf,
    i.e. a certain reject, rather than poisoning the chain.
    """
    delta = initial_energy - new_energy
    return torch.where(torch.isnan(delta), torch.full_like(delta, -math.inf),
                       delta)


def static_binomial_sampling(log_p_accept):
    """blackjax.mcmc.proposal.static_binomial_sampling, per chain.

    Returns (accepted (P,) bool, p_accept (P,)). blackjax draws
    `jax.random.bernoulli(key, p_accept)`; `rand() < p` is the same law.
    """
    p_accept = log_p_accept.exp().clamp(max=1.0)
    accepted = torch.rand_like(p_accept) < p_accept
    return accepted, p_accept


def _sumsq(x):
    """Per-chain sum of squares over all non-chain dimensions."""
    return (x * x).flatten(1).sum(1)


def _precond_views(state, scale):
    """Work in y = x / s, where a preconditioned kernel is an ORDINARY one.

    Preconditioning with P = diag(s^2) is exactly a change of variables y = x/s:
        grad_y log pi~(y) = s * grad_x log pi(x)
    and the standard move y' = y + eps*g_y + sqrt(2 eps) xi maps back to
        x' = x + eps*s^2*grad_x + sqrt(2 eps)*s*xi,
    i.e. the usual preconditioned Langevin proposal. Doing it this way means the
    Metropolis ratio is the unpreconditioned one computed on (y, g_y), so it
    cannot be got wrong -- and the change-of-variables Jacobian is constant
    within a step, so it cancels in the ratio.

    Why this matters here: the target mixes a unit-Hessian Gaussian prior with a
    world-model cost whose curvature is badly conditioned, so a single isotropic
    step size is throttled by the stiffest direction. Inside SMC the population
    itself estimates the target covariance for free, which is the adaptive tuning
    of Buchholz, Chopin & Jacob (2021) (the scheme AFT cites for HMC-in-SMC).
    Returns (y, grad_y). scale=None is the identity and changes nothing.
    """
    if scale is None:
        return state.position, state.grad
    return state.position / scale, state.grad * scale


def _move_diag(drift, noise, displacement):
    """Is a proposed move DIRECTED or SCATTERED?

    A Langevin proposal is `a + eps*grad log pi + sqrt(2 eps)*xi`. The first term
    is drift (informative), the second diffusion (not). Their norm ratio is the
    move's signal-to-noise: below 1 the particle is mostly being scattered, and
    then a Metropolis filter has little signal to select on -- which is why a
    HIGHER acceptance rate can correlate with better planning. Rejection only
    helps when the drift it is filtering points somewhere useful.
    """
    dn = drift.flatten(1).norm(dim=1)
    nn = noise.flatten(1).norm(dim=1)
    return dict(drift_norm=float(dn.mean()),
                noise_norm=float(nn.mean()),
                move_snr=float((dn / nn.clamp_min(1e-12)).mean()),
                step_norm=float(displacement.flatten(1).norm(dim=1).mean()))


# ------------------------------------------------------------- ULA / JALA


def jarzynski_log_weight_increment(state, new, eps):
    """The weight correction a non-invariant ULA move needs. (P,)

    This is what makes ULA usable inside an SMC sampler, and it is exactly
    JALA (Carbone et al. Proposition 1, as used by JALA-EM): Del Moral et
    al. (2006) Eq. 12 with the backward kernel chosen to be the forward one,
    L_k(x', x) = M_k(x', x). The increment is then

        d logw = log pi(x') - log pi(x) + log M(x' -> x) - log M(x -> x')

    with M(x -> x') = N(x'; x + eps * grad log pi(x), 2 eps I).

    JALA writes the same quantity as A_{k+1} = A_k - alpha(x', x) + alpha(x, x')
    with alpha(x_l, x_r) = U(x_l) + (x_r - x_l).grad U(x_l)/2 + eps|grad U|^2/4
    and U = -log pi; expanding the Gaussians shows the two forms are identical,
    which `test_mcmc_kernels.test_jarzynski` checks numerically.

    For an MH-corrected kernel this term is zero, because the accept step makes
    the kernel exactly pi-invariant and the standard backward choice
    L(x', x) = pi(x) M(x, x') / pi(x') cancels it. That is the one structural
    advantage of paying for the accept step here: the weights need no
    correction, so nothing about them is approximate.
    """
    d = new.position - state.position
    log_fwd = -_sumsq(d - eps * state.grad) / (4.0 * eps)
    log_bwd = -_sumsq(-d - eps * new.grad) / (4.0 * eps)
    return (new.logdensity - state.logdensity) + log_bwd - log_fwd


class ULA:
    """Unadjusted Langevin. blackjax.mcmc.diffusions.overdamped_langevin.

    No Metropolis correction, so the kernel is not pi-invariant at any finite
    step size. Inside an SMC sampler that is not fatal but it is not free
    either: the incremental weight has to carry the Jarzynski correction above,
    which is precisely the JALA scheme. Reported through `log_weight_increment`,
    which the MH kernels return as zero.
    """

    name = "ula"
    evals_per_step = 1
    needs_grad = True

    def step(self, state, target, eps, scale=None, mu_after=None):
        """One ULA step; `mu_after` selects SOSMC's kernel indexing.

        With mu_after=None both kernels sit at the current target, so the
        increment below is the Jarzynski factor at a single potential and the
        change of target is left to the caller. With mu_after set, `state` is
        expected to be composed at theta_{k-1} and the proposal is recomposed at
        theta_k = mu_after, so the forward kernel is at theta_{k-1} and the
        backward at theta_k -- SOSMC Remark 1 -- and the increment becomes their
        Eq. (8) in full, equivalently the recursion of JALA-EM Eq. (7). The
        recompose reuses the cached C and grad C, so this costs no rollout.
        """
        y, gy = _precond_views(state, scale)
        noise = torch.randn_like(y)
        drift = eps * gy
        diffusion = math.sqrt(2.0 * eps) * noise
        y_prop = y + drift + diffusion
        prop = y_prop if scale is None else y_prop * scale
        new = target.evaluate(prop)
        if mu_after is not None:
            target.mu = mu_after
            new = target.recompose(new)
        y_new, gy_new = _precond_views(new, scale)
        ones = torch.ones(y.shape[0], device=y.device)
        # Jarzynski increment in the kernel's OWN (preconditioned) coordinates.
        inc = ((new.logdensity - state.logdensity)
               + (-_sumsq(-(y_new - y) - eps * gy_new) / (4.0 * eps))
               - (-_sumsq((y_new - y) - eps * gy) / (4.0 * eps)))
        return new, dict(
            acceptance_rate=ones, is_accepted=ones.bool(),
            log_weight_increment=inc,
            **_move_diag(drift, diffusion, y_prop - y))


# -------------------------------------------------------------------- MALA


def _mala_transition_energy(state_pos, new_pos, new_logdensity, new_grad, eps):
    """blackjax.mcmc.mala.build_kernel.transition_energy, verbatim.

        theta = x - y - eps * grad log pi(y)
        E(x -> y) = -log pi(y) + (1 / 4 eps) * ||theta||^2

    The acceptance ratio below is E(y -> x) - E(x -> y), which expands to the
    textbook MALA ratio
        log pi(y) - log pi(x) + log q(x|y) - log q(y|x)
    with q the Gaussian proposal density.
    """
    theta = state_pos - new_pos - eps * new_grad
    return -new_logdensity + 0.25 * (1.0 / eps) * _sumsq(theta)


class MALA:
    """Metropolis-adjusted Langevin. blackjax.mcmc.mala.build_kernel.

    One rollout per step: the proposal's (log pi, grad) become the current
    state's whenever the move is accepted, so the gradient evaluated at the
    rejected proposal is the only wasted work.
    """

    name = "mala"
    evals_per_step = 1
    needs_grad = True

    def step(self, state, target, eps, scale=None):
        y, gy = _precond_views(state, scale)
        noise = torch.randn_like(y)
        drift = eps * gy
        diffusion = math.sqrt(2.0 * eps) * noise
        y_prop = y + drift + diffusion
        prop = y_prop if scale is None else y_prop * scale
        new = target.evaluate(prop)
        y_new, gy_new = _precond_views(new, scale)
        diag = _move_diag(drift, diffusion, y_prop - y)

        new_energy = _mala_transition_energy(
            y, y_new, new.logdensity, gy_new, eps)
        prev_energy = _mala_transition_energy(
            y_new, y, state.logdensity, gy, eps)
        log_p_accept = safe_energy_diff(prev_energy, new_energy)

        accepted, p_accept = static_binomial_sampling(log_p_accept)
        return (new.where(accepted, state),
                dict(acceptance_rate=p_accept, is_accepted=accepted,
                     log_weight_increment=torch.zeros_like(p_accept), **diag))


# --------------------------------------------------------------------- RWM


class RWM:
    """Random-walk Metropolis. blackjax.mcmc.random_walk.build_rmh.

    Symmetric Gaussian proposal, so `build_rmh_transition_energy` with
    `proposal_logdensity_fn=None` reduces the transition energy to
    -log pi(y) and the ratio to log pi(y) - log pi(x).

    Gradient-free as a sampler, but `target.evaluate` returns the cost
    gradient anyway (one backward pass is cheap next to the rollout) so that
    the M-step still gets its pathwise gradient for free. That is bookkeeping,
    not part of the kernel: nothing in the accept decision below touches it.
    """

    name = "rwm"
    evals_per_step = 1
    needs_grad = False

    def step(self, state, target, eps, scale=None):
        y, _ = _precond_views(state, scale)
        noise = torch.randn_like(y)
        y_prop = y + eps * noise
        prop = y_prop if scale is None else y_prop * scale
        new = target.evaluate(prop)
        log_p_accept = safe_energy_diff(-state.logdensity, -new.logdensity)
        accepted, p_accept = static_binomial_sampling(log_p_accept)
        # RWM has no drift term at all, so its SNR is 0 by construction: every
        # move is diffusion and the accept step is the ONLY source of direction.
        diag = _move_diag(torch.zeros_like(noise), eps * noise, y_prop - y)
        return (new.where(accepted, state),
                dict(acceptance_rate=p_accept, is_accepted=accepted,
                     log_weight_increment=torch.zeros_like(p_accept), **diag))


# --------------------------------------------------------------------- HMC


class HMC:
    """HMC with an identity mass matrix and velocity-Verlet integration.

    blackjax.mcmc.hmc.build_kernel + blackjax.mcmc.integrators.velocity_verlet
    with kinetic energy 0.5 p'p, for which the momentum half-step is
    `p += 0.5*eps*grad log pi(q)` and the position step is `q += eps*p`.
    Hamiltonian H = -log pi(q) + 0.5 ||p||^2; the ratio is H_initial - H_final.
    The final momentum flip blackjax applies is omitted because the kinetic
    energy is symmetric in p, so it cannot change the ratio.

    Costs `n_leaps` rollouts per step, which is why n_leaps=1 is the default
    here: in a planner the kernel's rollout budget competes directly with the
    M-step's.
    """

    name = "hmc"
    needs_grad = True

    def __init__(self, n_leaps=1):
        self.n_leaps = int(n_leaps)

    @property
    def evals_per_step(self):
        return self.n_leaps

    def step(self, state, target, eps, scale=None):
        y0, gy = _precond_views(state, scale)
        p = torch.randn_like(y0)
        p0 = p.clone()
        h_initial = -state.logdensity + 0.5 * _sumsq(p)

        cur, y = state, y0
        for _ in range(self.n_leaps):
            p = p + 0.5 * eps * gy
            y = y + eps * p
            q = y if scale is None else y * scale
            cur = target.evaluate(q)
            _, gy = _precond_views(cur, scale)
            p = p + 0.5 * eps * gy

        h_final = -cur.logdensity + 0.5 * _sumsq(p)
        log_p_accept = safe_energy_diff(h_initial, h_final)
        accepted, p_accept = static_binomial_sampling(log_p_accept)
        # For a momentum method the per-leapfrog drift/noise split is meaningless:
        # the momentum is drawn ONCE and carried, so the trajectory is coherent by
        # construction and displacement grows like L*eps rather than sqrt(L)*eps
        # (Neal 2011, "avoiding random walks"). Report the whole-trajectory
        # displacement against the single momentum draw instead.
        disp = y - y0
        return (cur.where(accepted, state),
                dict(acceptance_rate=p_accept, is_accepted=accepted,
                     log_weight_increment=torch.zeros_like(p_accept),
                     **_move_diag(disp, math.sqrt(1.0) * p0, disp)))


# ------------------------------------------------------- step-size adaptation


class DualAveraging:
    """blackjax.adaptation.step_size.dual_averaging_adaptation (scalar).

    Nesterov dual averaging on log eps to hit a target acceptance rate
    (Hoffman & Gelman 2014, sec 3.2.1), with blackjax's defaults t0=10,
    gamma=0.05, kappa=0.75 and mu = log(10 * eps_0).

    Fed the population-mean acceptance rate, since the P chains share eps.
    `step_size` is the iterate to run with while adapting; `final()` is the
    averaged value to freeze at.
    """

    def __init__(self, target, eps0, t0=10, gamma=0.05, kappa=0.75):
        self.target = float(target)
        self.t0, self.gamma, self.kappa = t0, gamma, kappa
        self.mu = math.log(10.0 * eps0)
        self.step_size = float(eps0)
        self.log_step_size_avg = 0.0
        self.error_sum = 0.0
        self.t = 0

    def update(self, p_accept):
        self.t += 1
        eta_t = 1.0 / (self.t + self.t0)
        self.error_sum = ((1.0 - eta_t) * self.error_sum
                          + eta_t * (self.target - float(p_accept)))
        log_eps = self.mu - (math.sqrt(self.t) / self.gamma) * self.error_sum
        eta = self.t ** (-self.kappa)
        self.log_step_size_avg = (eta * log_eps
                                  + (1.0 - eta) * self.log_step_size_avg)
        self.step_size = math.exp(log_eps)
        return self.step_size

    def final(self):
        return math.exp(self.log_step_size_avg)


# ---------------------------------------------------------------- resampling


def systematic_resample(logw):
    """Systematic resampling (Douc & Cappe). Same routine as fk_planner's."""
    w = torch.softmax(logw, dim=0)
    n = w.shape[0]
    positions = (torch.rand(1, device=w.device)
                 + torch.arange(n, device=w.device, dtype=torch.float32)) / n
    cumsum = torch.cumsum(w, dim=0)
    cumsum[-1] = 1.0
    return torch.searchsorted(cumsum, positions).clamp(max=n - 1)


def ess_of(logw):
    """Effective sample size 1/sum(w^2) of the normalised weights."""
    w = torch.softmax(logw, dim=0)
    return 1.0 / (w * w).sum().item()


def selection_kl(logw):
    """KL(softmax(logw) || uniform) in nats: one resample's selection pressure."""
    n = logw.shape[0]
    w = torch.softmax(logw, dim=0)
    return float((w * (torch.log(w.clamp_min(1e-30)) + math.log(n))).sum().item())


# ------------------------------------------------------------------ registry


_KERNELS = {"ula": ULA, "mala": MALA, "rwm": RWM, "hmc": HMC}


def make_kernel(name, hmc_leaps=1):
    name = name.lower()
    if name not in _KERNELS:
        raise ValueError(f"unknown kernel {name!r}; have {sorted(_KERNELS)}")
    return HMC(n_leaps=hmc_leaps) if name == "hmc" else _KERNELS[name]()
