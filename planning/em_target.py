"""The target log-density for the EM/SMC planner: what `log pi` actually is.

The FK planner (fk_planner.py) never had an explicit target. Its move was
`a <- a - lr*Adam(grad L) + sqrt(2*lr*T)*xi`, which is a Langevin
discretisation of `exp(-L/T)` only if the preconditioner is dropped, and the
tilt `exp(-beta*V)` was applied through the weights alone. Because the drift
came from an optimizer, no Metropolis ratio could be written down, so no
kernel in that file is pi-invariant and the AIS weight identity
(prop:weight in the paper) was assumed, not satisfied.

This module writes the target down instead. With the path-integral /
MPPI-as-EM reading (Okada & Taniguchi 2018; Williams et al. 2017: MPPI is EM
for a Gaussian proposal whose mean is the M-step parameter), the E-step
target at SMC iteration k is

    log pi_{mu,beta}(a) = log q(a | mu) - beta * C(a) / alpha + const
                        = -||a - mu||^2 / (2 sigma_p^2) - beta * C(a) / alpha

    C(a) = objective_fn(rollout(a), z_g) + lambda_v * V(z_H(a))
                                         + action_reg * mean(a^2)

where

  mu        (H, A)  the M-step parameter: the shared nominal action sequence.
                    This is MPPI's mean, and the only thing the optimizer
                    touches. The kernel never updates it.
  sigma_p           the proposal/prior width. q(.|mu) = N(mu, sigma_p^2 I) is
                    a proper density, so pi is normalisable for any C, and at
                    beta = 0 the target IS the prior -- particles can be drawn
                    from it exactly (i.i.d.), which is a correct SMC
                    initialisation rather than fk_planner's
                    `init + N(0, init_std^2)` patch for its clone-degeneracy
                    bug.
  alpha             the path-integral temperature. beta*C/alpha is the only
                    place the cost enters, so C's units are absorbed once,
                    here, instead of being split between a Langevin
                    temperature and a tilt exponent.
  beta      [0, 1]  the tempering parameter. beta = 1 is the actual target;
                    the SMC path anneals 0 -> 1, so gamma_K = pi exactly
                    (the beta_K = 1 condition the paper's propositions need).
  lambda_v          weight on the roadmap cost-to-go inside C. V enters the
                    TARGET here, not only the weights: with a pi-invariant
                    kernel that is the consistent thing to do, because the
                    kernel and the weights must refer to the same pi. Set
                    lambda_v = 0 to recover a pure L2 target.

Both terms are differentiable: the prior analytically, C through the frozen
world model by autograd. So grad log pi is available, which is what MALA /
HMC / ULA need, and the Metropolis ratio is well defined.

Two deliberate omissions, because they would silently break the sampler:

  * No mollification (num_smooth_samples) inside C. A Monte-Carlo-noisy
    log pi makes every Metropolis ratio invalid. Gaussian loss smoothing is
    an OPTIMIZER device (it is what gd_reacher's 93% came from), so in this
    design it belongs to the M-step, where there is no accept step -- see
    `EMSMCPlanner`'s m_smooth_samples. Separating E from M is what makes it
    possible to keep both.
  * No hard action clamp inside the kernel. Projecting a proposal onto a box
    is not a reversible move, so the clamp would void the MH ratio. Action
    magnitude is controlled by the Gaussian prior (sigma_p) and by
    action_reg inside C, both of which are part of pi.

action_reg is FIXED, not annealed as in gd.py/fk_planner.py. An annealed
coefficient makes pi non-stationary in a second way, and every such change
has to appear in the incremental weight; beta already provides one annealed
direction and mu provides the other (both are tracked in the weight, see
EMSMCPlanner._log_incremental_weight).
"""

import numpy as np
import torch

from .mcmc_kernels import ChainState


class ActionSequenceTarget:
    """log pi_{mu,beta} over action sequences, and its gradient.

    Batched over particles: every call takes (P, H, A) and issues exactly ONE
    world-model rollout, so the sampler's cost is one rollout per kernel step.
    """

    def __init__(self, wm, objective_fn, get_latent, trans_obs_0, z_obs_g,
                 heat_solver=None, horizon=None, action_dim=None,
                 sigma_p=1.0, alpha=1.0, lambda_v=0.0, action_reg=0.0,
                 eval_chunk=0, device=None, v_traj="final", v_base=2.0):
        self.wm = wm
        self.objective_fn = objective_fn
        self.get_latent = get_latent
        self.heat_solver = heat_solver
        self.horizon = horizon
        self.action_dim = action_dim
        self.sigma_p = float(sigma_p)
        self.alpha = float(alpha)
        self.lambda_v = float(lambda_v)
        # Roadmap cost over the rollout. "final": Vhat(z_T). "all": every
        # predicted state, weighted base^s / sum base^s (s = 0..H-1), the
        # frame weighting of the DINO-WM all-frames objective
        # (objectives.objective_fn_all): the whole path is scored, the final
        # state most.
        assert v_traj in ("final", "all"), v_traj
        self.v_traj = v_traj
        self.v_base = float(v_base)
        self.action_reg = float(action_reg)
        # Max sequences per rollout inside evaluate(). Each sequence rolls out
        # independently and its gradient depends only on its own row, so
        # chunking is EXACT -- it only caps peak memory. Needed to reach
        # populations above ~24 at all: measured ~0.5 GB per sequence with a
        # live autograd graph, so P=32 alone would exceed a 16 GB card.
        self.eval_chunk = int(eval_chunk)
        self.device = device

        # Un-batched start/goal; expanded on demand and cached per batch size,
        # since the M-step may evaluate S*P sequences while the kernel
        # evaluates P.
        self._obs_0 = trans_obs_0
        self._z_g = z_obs_g
        self._obs_cache, self._zg_cache = {}, {}

        # The two moving parts of the target. mu is the M-step parameter and
        # beta the tempering level; both are set by the planner, never here.
        self.mu = None            # (H, A) tensor
        self.beta = 0.0

        # Budget accounting. The paper has to state each planner's rollout
        # count (iterations are matched across planners, rollouts are not),
        # so count them at the only place they happen.
        self.n_rollout_calls = 0
        self.n_rollout_seqs = 0

    # ---------------------------------------------------------------- batching

    def _expand(self, n):
        if n not in self._obs_cache:
            def rep(v):
                return v[0].unsqueeze(0).expand(n, *v.shape[1:]).contiguous()
            self._obs_cache[n] = ({k: rep(v) for k, v in self._obs_0.items()}
                                  if isinstance(self._obs_0, dict) else rep(self._obs_0))
            self._zg_cache[n] = {k: rep(v).detach() for k, v in self._z_g.items()}
        return self._obs_cache[n], self._zg_cache[n]

    # -------------------------------------------------------------------- cost

    def cost(self, a, components=False):
        """C(a) for a batch of action sequences. (P, H, A) -> (P,)

        One rollout per call. Differentiable: the caller decides whether to
        take a gradient. With `components=True` also returns the task term and
        the roadmap value separately -- the task term is what selection and the
        success logs are reported against (matching fk_planner's argmin L_dyn),
        and V is the tilt diagnostic.
        """
        n = a.shape[0]
        obs, zg = self._expand(n)
        i_z, _ = self.wm.rollout(obs_0=obs, act=a)
        self.n_rollout_calls += 1
        self.n_rollout_seqs += n
        task = self.objective_fn(i_z, zg, a)
        C = task
        V = None
        if self.heat_solver is not None:
            lat = self.get_latent(i_z)
            if self.v_traj == "all":
                seq = lat[:, -self.horizon:, :]          # predicted states only
                Tn = seq.shape[1]
                c = self.v_base ** torch.arange(Tn, device=seq.device, dtype=seq.dtype)
                c = c / c.sum()
                Vs = self.heat_solver.evaluate_rollout(seq.reshape(n * Tn, -1)).view(n, Tn)
                V = (Vs * c).sum(dim=1)
            else:
                V = self.heat_solver.evaluate_rollout(lat[:, -1, :])
            if self.lambda_v != 0.0:
                C = C + self.lambda_v * V
        if self.action_reg > 0:
            C = C + self.action_reg * (a ** 2).mean(dim=(1, 2))
        return (C, task, V) if components else C

    # ------------------------------------------------------------- log density

    def log_prior(self, a, mu=None):
        """log q(a | mu) up to the (mu-independent) normalising constant."""
        mu = self.mu if mu is None else mu
        return -((a - mu) ** 2).flatten(1).sum(1) / (2.0 * self.sigma_p ** 2)

    def _compose(self, a, C, gC, task=None, V=None):
        """Assemble log pi and grad log pi from the cost parts. No rollout.

        This is the factorisation the whole design rests on: mu and beta enter
        log pi only through closed-form terms,

            log pi      = -||a - mu||^2 / (2 sigma_p^2) - beta * C / alpha
            grad log pi = -(a - mu) / sigma_p^2         - beta * gC / alpha

        so once C and grad C are cached at a position, ANY change to mu (the
        M-step) or beta (the tempering) can be applied to the chain state
        exactly and for free. Only moving the particles costs a rollout, which
        is what keeps the kernel's budget and the optimiser's budget separate
        and separately accountable.
        """
        logp = self.log_prior(a) - self.beta * C / self.alpha
        g = -(a - self.mu) / self.sigma_p ** 2 - self.beta * gC / self.alpha
        return ChainState(position=a, logdensity=logp, grad=g,
                          cost=C, cost_grad=gC, task=task, value=V)

    def evaluate(self, a):
        """ONE rollout -> a fully populated ChainState at `a`.

        The single entry point the samplers use. The cost gradient is always
        taken, even for the gradient-free kernel (RWM): one backward pass is
        cheap next to the rollout, and caching grad C is what lets the M-step
        reuse the kernel's rollout instead of paying for its own.
        """
        n = a.shape[0]
        chunk = self.eval_chunk if 0 < self.eval_chunk < n else n
        if chunk >= n:
            a = a.detach().requires_grad_(True)
            C, task, V = self.cost(a, components=True)
            gC = torch.autograd.grad(C.sum(), a)[0]
            return self._compose(a.detach(), C.detach(), gC.detach(),
                                 task=task.detach(),
                                 V=None if V is None else V.detach())
        Cs, tasks, Vs, gCs = [], [], [], []
        for st in range(0, n, chunk):
            ab = a[st:st + chunk].detach().requires_grad_(True)
            C, task, V = self.cost(ab, components=True)
            gCs.append(torch.autograd.grad(C.sum(), ab)[0].detach())
            Cs.append(C.detach())
            tasks.append(task.detach())
            Vs.append(None if V is None else V.detach())
        return self._compose(
            a.detach(), torch.cat(Cs), torch.cat(gCs),
            task=torch.cat(tasks),
            V=None if Vs[0] is None else torch.cat(Vs))

    def recompose(self, state):
        """Rebuild log pi / grad log pi at the current (mu, beta). No rollout.

        Call after the planner changes either. The particles do not move, so
        the cached C and grad C stay valid.
        """
        return self._compose(state.position, state.cost, state.cost_grad,
                             task=state.task, V=state.value)

    def log_prob(self, a):
        with torch.no_grad():
            return self.log_prior(a) - self.beta * self.cost(a) / self.alpha

    def log_prob_and_grad(self, a):
        """((P,), (P, H, A), (P,)) = (log pi, grad log pi, C). One rollout."""
        s = self.evaluate(a)
        return s.logdensity, s.grad, s.cost

    # --------------------------------------------------------------- utilities

    def sample_prior(self, n, generator=None):
        """Exact i.i.d. draw from pi_{mu, beta=0} = N(mu, sigma_p^2 I)."""
        eps = torch.randn(n, *self.mu.shape, device=self.mu.device,
                          dtype=self.mu.dtype, generator=generator)
        return self.mu.unsqueeze(0) + self.sigma_p * eps

    def rollout_stats(self):
        return dict(rollout_calls=self.n_rollout_calls,
                    rollout_seqs=self.n_rollout_seqs)
