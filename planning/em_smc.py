"""EM planner: SMC over action sequences (E-step) + Adam on the nominal
sequence (M-step), with the two kept strictly apart.

This is the fix for what went wrong in fk_planner.py. There, one update did
both jobs at once,

    a <- a - lr * Adam(grad L) + sqrt(2 * lr * T) * xi

so the "kernel" was an optimiser: preconditioned, momentum-carrying,
descending an annealed objective. It is not pi-invariant for any pi that can be
written down, which means no Metropolis ratio exists for it and the AIS weight
identity was assumed rather than satisfied. The kernel ablations that replaced
it with a real sampler (kernel=mala/hmc/sgld in fk_planner) then failed for the
opposite reason: swapping the optimiser out for a sampler deleted the descent
that was doing the work. Neither half survives on its own.

JALA-EM's structure is the way out, and it is also MPPI's. Keep two variables,
not one:

  mu  (H, A)     the nominal action sequence. The M-step parameter. Only the
                 optimiser touches it.
  a   (P, H, A)  the particles. Only the kernel touches them.

and alternate

  E-step   run a pi_{mu,beta}-invariant MCMC kernel on the particles and
           reweight them (planning/mcmc_kernels.py: MALA, RWM, HMC, ULA,
           ported from blackjax). The kernel does no optimisation: no
           preconditioner, no momentum, no annealed objective, no loss
           smoothing inside the accept step.

  M-step   take an Adam step on mu along an SMC-weight-driven gradient. The
           weights from the E-step are the only channel by which the sampler
           influences the descent, which is what makes this EM rather than two
           heuristics running side by side.

The target is planning/em_target.py:

    log pi_{mu,beta}(a) = -||a - mu||^2 / (2 sigma_p^2) - beta * C(a) / alpha

Both M-step gradients below estimate the SAME quantity, grad_mu(-log Z(mu)) --
they are its score-function and reparameterised forms, not two complementary
terms. Verified numerically (test_m_step_equivalence): cosine +0.9999 and
|g_path| / |g_score| = alpha exactly. So `m_step="both"` MIXES them convexly;
adding them, as an earlier version did, double-counts the gradient at a
magnitude that varies with alpha, and with alpha="auto" that varied per seed.

  score     g = -sum_i w_i (a_i - mu) / sigma_p^2
            Since d/dmu log Z = E_pi[d/dmu log q(a|mu)]. Setting it to zero
            gives mu = sum_i w_i a_i, so a single plain-SGD step of size
            sigma_p^2 along this gradient IS MPPI's weighted-mean update; Adam
            turns that closed-form jump into a damped, preconditioned one.
            No world-model gradient appears.

  pathwise  g = sum_i w_i grad_a C(a_i) / alpha
            Substituting a = mu + sigma_p * eps into Z(mu) and differentiating
            gives grad_mu(-log Z) = E_pi[grad C(a)] / alpha; the 1/alpha is what
            puts it on the same scale as the score form. This is where the world
            model's Jacobian enters, i.e. where GRASP lives, and it is a
            Monte-Carlo estimate of grad E_{a~pi}[C(a)] -- the expectation form
            that puts this in the SOSMC family.

Because both are unbiased for the same gradient, any convex combination is too,
and the mix only trades their variances. Under Adam a constant rescaling of the
total gradient is invisible (verified to ~1e-7), so `score` and `pathwise` on
their own are insensitive to alpha; only the mix depends on getting it right.

Both come for free from the kernel's own rollout, because ChainState caches
grad C and mu enters log pi only through the prior (em_target._compose). With
m_smooth_samples > 1 the pathwise term is instead estimated at Gaussian-noised
copies of each particle, which is GRASP's Corollary 1 estimator
(E_xi[grad C(a + xi)] = grad C_sigma(a)) and the device behind gd_reacher's
93%. That smoothing is legitimate HERE and nowhere else: the M-step has no
accept step for a Monte-Carlo-noisy objective to invalidate. Separating E from
M is what makes it possible to keep both it and a correct sampler.

Weight bookkeeping is the textbook SMC sampler (Del Moral, Doucet & Jasra),
in the order reweight -> resample -> mutate. With a pi_k-invariant kernel the
mutation contributes nothing to the weights, and the incremental weight is the
new target over the old one evaluated at the current particles,

    d logw = [log q(a|mu_k) - log q(a|mu_{k-1})] - (beta_k - beta_{k-1}) C(a)/alpha

which is exact and costs no rollouts: the first bracket is analytic and C(a) is
already cached. Unlike fk_planner's telescoped anchor form, this needs no
assumption about the kernel, because the kernel actually is invariant.
"""

import math
import os
import time

import numpy as np
import torch
from einops import repeat

from .gd import GDPlanner
from .em_target import ActionSequenceTarget
from .mcmc_kernels import (DualAveraging, ess_of, make_kernel, selection_kl,
                           systematic_resample)


# Optimal-scaling acceptance targets: Roberts & Rosenthal (2001) for RWM and
# MALA, Beskos et al. (2013) for HMC. ULA has no accept step.
DEFAULT_TARGET_ACCEPT = {"rwm": 0.234, "mala": 0.574, "hmc": 0.651, "ula": None}


def bump_weights(logw, dlogw, log_zhat):
    """Apply an incremental weight and accumulate the SMC log-Z estimate.

    Standard SMC-sampler identity (Del Moral et al. 2006):
        log Zhat = sum_k log( sum_i W_k^i w_k^i )
    with W the normalised weights before the increment. In log space each term
    is logsumexp(logw + dlogw) - logsumexp(logw), which telescopes and is
    unaffected by the resampling resets, since resampling replaces a weighted
    sample by an unweighted one of the same total mass.

    WARNING -- this is NOT a valid coverage metric for THIS planner, and must
    not be reported as one. The accumulator is correct for a prescribed target
    path (validated against a closed-form log Z to within 0.02, and invariant to
    resampling frequency), but here mu_k is fit TO the particles by the M-step,
    so every mu-increment log q(a|mu_new) - log q(a|mu_old) is positive by
    construction. Adaptive-target SMC has no unbiased log Z estimate, and the
    bias here is overwhelming: since C >= 0 we have Z = E_q[exp(-C/alpha)] <= 1,
    hence log[Z_K/Z_0] <= |log Z_0| ~ mean(C)/alpha, which was ~1.8 on measured
    runs while this accumulator reported ~130. It is tracking how far mu
    travelled toward its own particles.

    Use `softmin_cost` below for coverage instead; it is path-free and so cannot
    drift this way. This value is kept only as a diagnostic of target motion.
    """
    new = logw + dlogw
    inc = float(torch.logsumexp(new, 0) - torch.logsumexp(logw, 0))
    return new, log_zhat + inc


def softmin_cost(C, alpha):
    """Plug-in soft-minimum of the cost over the current population. (scalar)

        -alpha * log( (1/P) sum_i exp(-C_i / alpha) )

    This is the coverage metric to report. It is the population estimate of
    -alpha*log Z(mu), i.e. a smooth surrogate for the best cost available in the
    region the proposal currently covers, and it tends to min_i C_i as
    alpha -> 0. Unlike the accumulated log Zhat it is computed fresh from the
    current particles, so the M-step fitting mu to those particles cannot inflate
    it across the run, and it is bounded by the actual cost scale.

    Lower is better, and it is directly comparable across kernels because
    nothing about it depends on resampling rate or path history.

    CHOOSE ALPHA CAREFULLY. At the planner's own alpha (= std(C) under
    alpha="auto") this is dominated by the population bulk and is effectively a
    mean, which is the WRONG functional for planning: the planner executes one
    sequence, the argmin, so coverage has to be tail-sensitive. Measured on
    seed 19, MALA had the better softminC at every step and failed while HMC had
    the worse softminC and succeeded, because HMC's single best particle was at
    0.53 against MALA's 1.13 at the deciding step. Rank kernels on `Cmin`, or on
    this at a cold alpha (0.1 * alpha is logged as softminCold); the planner's
    alpha is a SAMPLING temperature, not a selection one.
    """
    P = C.shape[0]
    return float(-alpha * (torch.logsumexp(-C / alpha, 0) - math.log(P)))


def adapt_beta(logw, U, beta_lo, beta_max, ess_target, n_bisect=40):
    """Largest beta in [beta_lo, beta_max] whose reweighting keeps ESS >= target.

    Adaptive tempering (Del Moral et al.): advance the tilt as fast as the
    population can pay for it, where `U = C / alpha` is the potential being
    tempered. Returns beta_lo unchanged when even that is unaffordable -- the
    caller then resamples and tries again next iteration.
    """
    if ess_of(logw - (beta_max - beta_lo) * U) >= ess_target:
        return beta_max
    lo, hi = beta_lo, beta_max
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if ess_of(logw - (mid - beta_lo) * U) >= ess_target:
            lo = mid
        else:
            hi = mid
    return lo


class EMSMCPlanner(GDPlanner):
    """Expectation-maximisation planner: SMC E-step, Adam M-step.

    Reuses GDPlanner's __init__ (replay-buffer loading, heat-solver/graph
    compilation, _get_latent, init_actions) and overrides plan(). Plans one
    env at a time, like FKSMCPlanner and WMRewardPlanner.
    """

    def __init__(
        self,
        # --- population and target ---
        num_particles=32,
        sigma_p=1.0,
        alpha=1.0,
        alpha_scale=1.0,
        lambda_v=0.0,
        v_traj="final",
        v_base=2.0,
        beta_max=1.0,
        beta_schedule="adaptive",
        beta_init=1.0,
        # --- E-step ---
        kernel="mala",
        n_kernel_steps=1,
        kernel_step_size=0.1,
        adapt_step_size=True,
        kernel_warmup=30,
        target_accept=None,
        hmc_leaps=1,
        hmc_jitter=True,
        precond="population",
        precond_floor=0.05,
        precond_ema=0.7,
        precond_power=0.5,
        ess_target_frac=0.5,
        resample_frac=0.5,
        # --- M-step ---
        m_step="score",
        m_lr=0.1,
        m_smooth_samples=1,
        m_smooth_sigma=1.0,
        m_chunk=8,
        eval_chunk=8,
        pathwise_mix=0.5,
        # --- selection ---
        select="task",
        stop_on_success=True,
        **kwargs,
    ):
        # GDPlanner wants these; the EM planner does not use its sampling or
        # its annealed regulariser (see the em_target docstring on why
        # action_reg must stay fixed here).
        kwargs.setdefault("sample_type", "zero")
        kwargs.setdefault("lr", m_lr)
        kwargs.setdefault("var_actions", 0.0)
        kwargs.setdefault("num_samples", 1)
        super().__init__(**kwargs)

        self.num_particles = num_particles
        self.sigma_p = sigma_p
        # alpha is the path-integral temperature: beta*C/alpha is the only
        # place the cost enters, so alpha absorbs C's units once. "auto" sets
        # it to the spread of C over the initial prior population, which makes
        # beta = 1 a ~1-nat tilt regardless of the environment's cost scale --
        # the usual adaptive-temperature choice in MPPI implementations. It is
        # then FROZEN, for the same reason action_reg is (a moving alpha makes
        # pi non-stationary in a way the incremental weight does not track).
        self.alpha = alpha
        # Temperature multiplier on alpha="auto": alpha = alpha_scale * std(C).
        # alpha_scale < 1 is a colder target (MPPI's lambda), which scales up the
        # cost term of the Langevin drift relative to its sqrt(2 eta) noise.
        self.alpha_scale = float(alpha_scale)
        self.lambda_v = lambda_v
        self.v_traj = v_traj
        self.v_base = v_base
        self.beta_max = beta_max
        # "adaptive" advances beta by the ESS bisection (Del Moral et al.).
        # That is the right control at large P, and the WRONG one at P=3: when
        # the population collapses to clones, C has no spread across particles,
        # so ESS reads a perfect P even though the population is degenerate, and
        # the bisection jumps straight to beta_max. beta then outruns the step
        # size -- grad log pi carries beta/alpha -- the acceptance rate goes to
        # zero, and nothing can re-diversify the clones (measured on seed 20:
        # beta 1 -> 1000 in 4 steps, acc 0.000, eps 1e-5, cumKL frozen for the
        # remaining 146 steps).
        # "geometric" instead raises beta log-linearly from beta_init to
        # beta_max across the run, which decouples the two adaptations: at
        # beta_max/beta_init = 1000 over 150 steps each step is a 4.7% increase,
        # well within what dual averaging tracks.
        assert beta_schedule in ("adaptive", "geometric"), beta_schedule
        self.beta_schedule = beta_schedule
        self.beta_init = beta_init

        assert kernel in ("mala", "rwm", "hmc", "ula"), kernel
        self.kernel = kernel
        self.n_kernel_steps = n_kernel_steps
        self.kernel_step_size = kernel_step_size
        self.adapt_step_size = adapt_step_size
        # Retained for config compatibility; no longer used. Freezing the step
        # size is wrong once beta anneals upward (see the E-step below).
        self.kernel_warmup = kernel_warmup
        self.target_accept = (DEFAULT_TARGET_ACCEPT[kernel]
                              if target_accept is None else target_accept)
        self.hmc_leaps = hmc_leaps
        # Randomise trajectory length. For a locally-Gaussian target the
        # Hamiltonian flow is harmonic, so displacement goes as 2|sin(eps*L/2)|:
        # it peaks at eps*L = pi and returns to the START at eps*L = 2*pi.
        # A FIXED L therefore resonates; Neal (2011) and Stan both jitter it.
        self.hmc_jitter = hmc_jitter
        # Metric adaptation. The target mixes a unit-Hessian Gaussian prior with a
        # badly-conditioned world-model cost, so one isotropic step size is
        # throttled by the stiffest direction -- measured on an ill-conditioned
        # Gaussian, an identity-metric MALA is held to eps=0.0016 where a
        # preconditioned one runs at 0.645 (416x) at the SAME acceptance, and is
        # more accurate. SMC gives the metric for free: the population estimates
        # the target covariance. This is the adaptive HMC-in-SMC tuning of
        # Buchholz, Chopin & Jacob (2021), which AFT cites.
        # NOTE the estimate uses the PREVIOUS iteration's population, so the
        # kernel depends on the particle history -- standard adaptive SMC, and the
        # reason it does not void the MH ratio within a step.
        # "grad" is the pSGLD preconditioner (Li et al., AAAI 2016): a diagonal
        # metric from an EMA of the COST gradient's second moments. Needed because
        # "population" is a near no-op here -- the isotropic prior dominates the
        # target (measured grad norms: prior 6.2 vs cost 2.9), so it keeps the
        # population isotropic and its covariance measures the prior, not the
        # curvature we need to fix (measured kappa only 1.0 -> 2.4). This is also
        # what FK's Adam-on-the-particles was supplying.
        assert precond in ("none", "population", "grad"), precond
        self.precond = precond
        self.precond_floor = precond_floor
        self.precond_ema = precond_ema
        # s = (floor + v^power)^(-1/2) on the cost-gradient second moment v.
        # power=0.5 gives s ~ v^(-1/4): that is Adam / RMSProp / pSGLD, which is
        # deliberately square-root damped and so converts a 64x gradient spread
        # into only ~2.8x of metric. power=1.0 gives s ~ v^(-1/2), the whitening
        # metric for a locally quadratic cost, which uses the full anisotropy.
        self.precond_power = precond_power
        self.ess_target_frac = ess_target_frac
        self.resample_frac = resample_frac

        assert m_step in ("both", "pathwise", "score", "none"), m_step
        self.m_step = m_step
        self.m_lr = m_lr
        self.m_smooth_samples = m_smooth_samples
        self.m_smooth_sigma = m_smooth_sigma
        # Max sequences per M-step rollout. P*S sequences with a live autograd
        # graph does not fit on a 16 GB card (measured: P=3, S=8, H=5 peaked at
        # 14.3 GiB and OOM'd), which is why the baseline sweeps had to run at
        # S=1. Chunking is exact here, not an approximation -- see
        # _smoothed_cost_grad.
        self.m_chunk = m_chunk
        # Same ceiling for the kernel's own rollout (see em_target.evaluate).
        self.eval_chunk = eval_chunk
        # Convex weight on the pathwise estimator when m_step="both":
        # g = (1 - mix) * g_score + mix * g_path. 0 = score only, 1 = pathwise
        # only. NOT a free scale -- the two estimate the same gradient.
        assert 0.0 <= pathwise_mix <= 1.0, pathwise_mix
        self.pathwise_mix = pathwise_mix

        assert select in ("task", "cost", "mu"), select
        self.select = select
        # Early exit on success CENSORS the run: successful runs stop at
        # iteration 0-13 while failures burn all 150, so any mid-run coverage
        # statistic compared between them silently drops the fast successes
        # (measured: Cmin@10 had 3 of 6 successes left against 8 of 8 failures).
        # Set False for measurement runs so every run yields a full coverage
        # trajectory plus an uncensored success label (`first_success_step`).
        self.stop_on_success = stop_on_success

    # ------------------------------------------------------------------ M-step

    def _m_grad(self, target, state, w, sigma_t):
        """An estimator of grad_mu(-log Z(mu)), handed to Adam. (H, A)

        `w` are the normalised SMC weights -- the only path from the E-step into
        the descent. The two forms below estimate the SAME gradient, so
        m_step="both" is a convex combination of them, never a sum.
        """
        wv = w.view(-1, 1, 1)

        # score form. A plain SGD step of size sigma_p^2 along this alone
        # reproduces MPPI exactly: mu <- sum_i w_i a_i.
        g_score = -((wv * (state.position - target.mu)).sum(0)
                    / self.sigma_p ** 2)
        if self.m_step == "score":
            return g_score

        # pathwise (reparameterised) form. The 1/alpha is what the derivation
        # requires and is what puts it on the score form's scale.
        if self.m_smooth_samples > 1 and sigma_t > 0:
            gC = self._smoothed_cost_grad(target, state.position, sigma_t)
        else:
            gC = state.cost_grad              # free: cached by the kernel
        g_path = (wv * gC).sum(0) / target.alpha
        if self.m_step == "pathwise":
            return g_path

        mix = self.pathwise_mix
        return (1.0 - mix) * g_score + mix * g_path

    def _smoothed_cost_grad(self, target, a, sigma):
        """E_xi[grad C(a + sigma xi)] per particle, by Monte Carlo. (P, H, A)

        GRASP's Corollary 1: the expected gradient at noised inputs is the
        gradient of the Gaussian-smoothed cost. One rollout of P * S sequences.
        Confined to the M-step on purpose -- a stochastic objective inside a
        Metropolis ratio would void it.
        """
        P, S = a.shape[0], self.m_smooth_samples
        y = (a.unsqueeze(1)
             + sigma * torch.randn(P, S, *a.shape[1:],
                                   device=a.device, dtype=a.dtype)).flatten(0, 1)
        # Row i of the gradient depends only on row i of y, so splitting the
        # P*S rollout into chunks and concatenating is EXACT -- it only caps
        # peak memory. Without it S > 1 is unaffordable (see m_chunk).
        chunk = self.m_chunk if self.m_chunk > 0 else y.shape[0]
        gs = []
        for st in range(0, y.shape[0], chunk):
            yb = y[st:st + chunk].detach().requires_grad_(True)
            C = target.cost(yb)
            gs.append(torch.autograd.grad(C.sum(), yb)[0].detach())
        return torch.cat(gs, 0).view(P, S, *a.shape[1:]).mean(1)

    # -------------------------------------------------------------------- plan

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(
            obs_0, obs_g, encode_to_vector=False)
        n_evals = (trans_obs_0["visual"].shape[0]
                   if isinstance(trans_obs_0, dict) else trans_obs_0.shape[0])
        assert n_evals == 1, "EMSMCPlanner plans one env at a time"

        init = self.init_actions(obs_0, actions).to(self.device)  # (1, H, A)
        P = P_ = self.num_particles
        ess_target = self.ess_target_frac * P
        resample_at = self.resample_frac * P

        # Fit the differentiable V surrogate for this goal (same offline-data
        # heat solver as gd.py / fk_planner.py).
        if self.heat_solver is not None:
            goal_latent = self._get_latent(z_obs_g)[0, 0].detach().cpu().numpy()
            self.heat_solver.solve_for_goal(goal_latent)

        target = ActionSequenceTarget(
            wm=self.wm, objective_fn=self.objective_fn,
            get_latent=self._get_latent, trans_obs_0=trans_obs_0,
            z_obs_g=z_obs_g, heat_solver=self.heat_solver,
            horizon=self.horizon, action_dim=self.action_dim,
            sigma_p=self.sigma_p,
            alpha=1.0 if self.alpha == "auto" else self.alpha,
            lambda_v=0.0 if self.lambda_v == "auto" else float(self.lambda_v),
            action_reg=self.action_reg, eval_chunk=self.eval_chunk,
            device=self.device, v_traj=self.v_traj, v_base=self.v_base)

        # mu is the M-step parameter and the only thing Adam owns.
        mu = init[0].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.m_lr)
        target.mu = mu.detach()
        mu_prev = target.mu.clone()

        # beta = 0 <=> the target IS the prior, so the population can be drawn
        # from it exactly. No clone-degeneracy patch needed (contrast
        # fk_planner's `init + N(0, init_std^2)` on an otherwise degenerate
        # population).
        target.beta = (0.0 if self.beta_schedule == "adaptive"
                       else self.beta_init)
        state = target.evaluate(target.sample_prior(P))
        if self.lambda_v == "auto" and state.value is not None:
            # Scale-match the roadmap term to the L2 term over the initial
            # population, so neither dominates C by units alone:
            # lambda_V = std(L) / std(Vhat). One extra rollout to re-evaluate C
            # (and grad C) with the V term in.
            s_l = float(state.task.std()); s_v = float(state.value.std())
            target.lambda_v = s_l / max(s_v, 1e-8)
            print(f"[EM-SMC] lambda_v=auto -> {target.lambda_v:.4g} "
                  f"(std L {s_l:.4g}, std V {s_v:.4g}, v_traj={target.v_traj})")
            state = target.evaluate(state.position)
        if self.alpha == "auto":
            target.alpha = max(self.alpha_scale * float(state.cost.std()), 1e-8)
            print(f"[EM-SMC] alpha=auto x{self.alpha_scale:g} -> {target.alpha:.4g} "
                  f"(C: mean {float(state.cost.mean()):.4g}, "
                  f"std {float(state.cost.std()):.4g})")
            # The state above was composed with the placeholder alpha=1, so its
            # log pi and grad log pi are stale. Free to fix (no rollout).
            state = target.recompose(state)
        logw = torch.zeros(P, device=self.device)

        kern = make_kernel(self.kernel, hmc_leaps=self.hmc_leaps)
        eps = self.kernel_step_size
        da = (DualAveraging(self.target_accept, eps)
              if self.adapt_step_size and self.target_accept is not None
              else None)

        # z->s fidelity probe (GRASP_EM_ORACLE=<path.npz>). Every
        # GRASP_EM_ORACLE_EVERY steps, evaluate EVERY particle in the real env,
        # not just the selected one. That separates the two failure modes the
        # SMC metrics cannot distinguish:
        #   coverage  -- does the population CONTAIN a sequence that succeeds?
        #                (the kernel's job; log Zhat is its soft-min surrogate,
        #                 but log Zhat lives in latent space and is blind to
        #                 p(s|z))
        #   selection -- given one exists, does argmin task loss pick it?
        #                (the proxy's job)
        # The rank correlation between the latent objective and the true
        # distance, computed across the population, is a direct estimate of the
        # z->s channel: if it is weak, no amount of coverage converts to
        # success and the kernel comparison is capped.
        _oracle = os.environ.get("GRASP_EM_ORACLE")
        _oracle_every = int(os.environ.get("GRASP_EM_ORACLE_EVERY", "25"))
        _orec = {"step": [], "task": [], "cost": [], "V": [], "success": [],
                 "d_true": [], "sel_idx": []} if _oracle else None

        exit_step = self.opt_steps
        # Accumulated log Z(mu_k, beta_k) / Z(mu_0, beta_0): how much better the
        # search region became than the one it started in. See bump_weights.
        log_zhat = 0.0
        best_ever_action, best_ever_task = None, math.inf
        first_success_step = None
        prev_disp = None
        scale = None
        grad_v = None
        kappa_gc = 1.0
        n_resamples, cum_kl, total_measured_time = 0, 0.0, 0.0
        current_success = 0.0
        mean_dist = mean_vis_dist = mean_proprio_dist = np.inf
        acc = 1.0

        for i in range(self.opt_steps):
            start = time.perf_counter()

            # -- 1. reweight for the previous M-step's move of mu. Exact and
            #       free: C(a) does not depend on mu, so the cost terms cancel
            #       and only the analytic prior ratio survives.
            if self.m_step != "none":
                logw, log_zhat = bump_weights(
                    logw,
                    target.log_prior(state.position, target.mu)
                    - target.log_prior(state.position, mu_prev),
                    log_zhat)
                state = target.recompose(state)

            # -- 2. reweight for the tempering increment, with beta advanced as
            #       far as the current population can afford.
            beta_prev = target.beta
            U = state.cost / target.alpha
            if self.beta_schedule == "geometric":
                f = (i + 1) / float(self.opt_steps)
                beta = self.beta_init * (self.beta_max
                                         / self.beta_init) ** f
            else:
                beta = adapt_beta(logw, U, beta_prev, self.beta_max, ess_target)
            if beta != beta_prev:
                logw, log_zhat = bump_weights(
                    logw, -(beta - beta_prev) * U, log_zhat)
                target.beta = beta
                state = target.recompose(state)

            # -- 3. resample on the ESS threshold.
            ess = ess_of(logw)
            resampled = ess < resample_at
            if resampled:
                cum_kl += selection_kl(logw)
                idx = systematic_resample(logw)
                state = state.index_select(idx)
                logw = torch.zeros(P, device=self.device)
                n_resamples += 1
                last_unique = int(torch.unique(idx).numel())
            else:
                last_unique = P

            # -- 4. E-step: pi_{mu,beta}-invariant mutation. No optimisation.
            # --- metric from the previous population, geometric-mean-normalised
            #     so eps keeps its scale and remains comparable across kernels ---
            with torch.no_grad():
                # anisotropy actually present in the cost gradient -- if this is
                # ~1 no diagonal preconditioner can help, whatever it is built from
                rms_gc = state.cost_grad.pow(2).mean(0).sqrt()
                kappa_gc = float(rms_gc.max() / rms_gc.clamp_min(1e-12).min())
            if self.precond == "population":
                with torch.no_grad():
                    sd = state.position.std(dim=0).clamp_min(self.precond_floor)
                    sd = sd / sd.log().mean().exp()
                    scale = (sd if scale is None else
                             self.precond_ema * scale + (1 - self.precond_ema) * sd)
            elif self.precond == "grad":
                with torch.no_grad():
                    v = rms_gc.pow(2)
                    v_ema = (v if grad_v is None else
                             self.precond_ema * grad_v + (1 - self.precond_ema) * v)
                    grad_v = v_ema
                    # pSGLD: G = diag(1/(lambda + sqrt(V))); s = sqrt(G)
                    sd = (self.precond_floor
                          + v_ema.pow(self.precond_power)).rsqrt()
                    scale = sd / sd.log().mean().exp()
            accs = []
            pos_before = state.position.clone()
            kdiag = {}
            for _ in range(self.n_kernel_steps):
                if self.kernel == "hmc" and self.hmc_jitter:
                    kern.n_leaps = int(torch.randint(
                        1, 2 * max(1, self.hmc_leaps) + 1, (1,)).item())
                state, info = kern.step(state, target, eps, scale=scale)
                kdiag = {k: v for k, v in info.items()
                         if k in ("drift_norm", "noise_norm", "move_snr",
                                  "step_norm")}
                # Zero for the MH-corrected kernels (they are pi-invariant, so
                # the mutation cannot change the weights); the Jarzynski
                # correction for ULA, which is what makes that case JALA.
                logw, log_zhat = bump_weights(
                    logw, info["log_weight_increment"], log_zhat)
                accs.append(float(info["acceptance_rate"].mean()))
            acc = float(np.mean(accs)) if accs else 1.0

            # An unadjusted kernel has no accept step to veto a bad move, so a
            # step size past the stability limit diverges outright rather than
            # stalling -- measured: ULA at eps=3.2 went non-finite at step 54 and
            # took the whole run down with an EOFError, losing the seed instead of
            # recording a failure. Detect it and exit the loop cleanly so the run
            # still writes its (failed) result.
            if not torch.isfinite(state.cost).all():
                print(f"[EM-SMC] DIVERGED at step {i}: non-finite cost "
                      f"({int((~torch.isfinite(state.cost)).sum())}/{P} particles). "
                      f"kernel={self.kernel} eps={eps:.4g} -- stopping cleanly.")
                self.diverged_at = i
                break
            if da is not None:
                # Adapt throughout, never freeze. grad log pi carries
                # -beta*grad C/alpha, so as the tempering drives beta up the
                # usable step size falls; a step size frozen after warmup
                # collapses the acceptance rate later in the run. Dual
                # averaging's sqrt(t)/gamma factor makes it progressively
                # calmer on its own, so the early thrash is a t-small
                # transient rather than something freezing has to fix.
                eps = da.update(acc)

            # --- is the population being DRIVEN or SCATTERED? ---
            # grad log pi = -(a - mu)/sigma_p^2 - beta*grad C/alpha. Split it, so
            # a drift that is mostly the prior pulling particles back to mu (and
            # therefore carrying no task information) is visible as such.
            with torch.no_grad():
                g_prior = -(state.position - target.mu) / self.sigma_p ** 2
                g_cost = -(target.beta / target.alpha) * state.cost_grad
                kdiag["grad_prior"] = float(g_prior.flatten(1).norm(dim=1).mean())
                kdiag["grad_cost"] = float(g_cost.flatten(1).norm(dim=1).mean())
                # Directional persistence: cos between this displacement and the
                # last one. ~0 means a random walk; >0 means directed motion.
                disp = state.position - pos_before
                if prev_disp is not None:
                    a1 = disp.flatten(1)
                    a0 = prev_disp.flatten(1)
                    den = (a1.norm(dim=1) * a0.norm(dim=1)).clamp_min(1e-12)
                    kdiag["cos_persist"] = float(((a1 * a0).sum(1) / den).mean())
                prev_disp = disp

            # -- 5. M-step: Adam along the SMC-weighted gradient.
            w = torch.softmax(logw, dim=0)
            if self.m_step != "none":
                sigma_t = self.m_smooth_sigma * max(
                    0.0, 1.0 - (i / float(self.opt_steps)))
                g = self._m_grad(target, state, w, sigma_t)
                mu_prev = target.mu.clone()
                mu.grad = g
                optimizer.step()
                target.mu = mu.detach().clone()

            total_measured_time += time.perf_counter() - start

            # -- 6. selection. argmin task loss over the particles, and over mu
            #       as well since mu is what MPPI would execute.
            task = state.task
            best_idx = int(torch.argmin(state.cost if self.select == "cost"
                                        else task))
            best_action = state.position[best_idx:best_idx + 1]
            best_task = float(task[best_idx])
            use_mu = self.select == "mu"
            if self.evaluator is not None and i % self.eval_every == 0:
                with torch.no_grad():
                    _, mu_task, _ = target.cost(target.mu.unsqueeze(0),
                                                components=True)
                if use_mu or float(mu_task[0]) < best_task:
                    best_action = target.mu.unsqueeze(0)
                    best_task = float(mu_task[0])
                    use_mu = True

            if self.action_clamp > 0:
                best_action = best_action.clamp(-self.action_clamp,
                                                self.action_clamp)
            if best_task < best_ever_task:
                best_ever_task = best_task
                best_ever_action = best_action.detach().clone()

            V_pop = (state.value if state.value is not None
                     else torch.zeros_like(task))
            delta_EV = float((w * V_pop).sum() - V_pop.mean())

            if i % self.eval_every == 0:
                print(f"[EM-SMC] step={i:4d} | beta={beta:.3f} "
                      f"| L_dyn={best_task:.4f}{'(mu)' if use_mu else ''} "
                      f"(best {best_ever_task:.4f}) "
                      f"| V_sel={float(V_pop[best_idx]):.4f} "
                      f"(pop {float(V_pop.mean()):.2f}) "
                      f"| ESS={ess:.1f}/{P} | dEV={delta_EV:+.3f} "
                      f"| resamples={n_resamples} (uniq={last_unique}) "
                      f"| cumKL={cum_kl:.2f} "
                      f"| {self.kernel} acc={acc:.3f} eps={eps:.4g} "
                      f"| snr={kdiag.get('move_snr', float('nan')):.3f} "
                      f"drift={kdiag.get('drift_norm', float('nan')):.3f} "
                      f"noise={kdiag.get('noise_norm', float('nan')):.3f} "
                      f"gP={kdiag.get('grad_prior', float('nan')):.2f} "
                      f"gC={kdiag.get('grad_cost', float('nan')):.2f} "
                      f"cosp={kdiag.get('cos_persist', float('nan')):+.3f} "
                      f"| kappa={(float(scale.max() / scale.min()) if scale is not None else 1.0):.1f} "
                      f"kgc={kappa_gc:.1f} "
                      f"| b/a={beta / target.alpha:.3g} "
                      f"| Cmin={float(state.cost.min()):.4f} "
                      f"| softminC={softmin_cost(state.cost, target.alpha):.4f} "
                      f"| softminCold={softmin_cost(state.cost, 0.1 * target.alpha):.4f} "
                      f"| logZ*={log_zhat:+.2f} "
                      f"| rollouts={target.n_rollout_seqs}")

            if _orec is not None and i % _oracle_every == 0:
                # One particle at a time: the vector env asserts
                # len(action) == num_envs and the planner runs with n_evals=1,
                # so candidates cannot be batched through the real env. With
                # batch 1, logs["mean_vis_dist"] IS that particle's true goal
                # distance (point maze defines success as vis_dist < 0.5).
                acts = state.position.detach()
                if self.action_clamp > 0:
                    acts = acts.clamp(-self.action_clamp, self.action_clamp)
                suc_all, d_all = [], []
                for j in range(acts.shape[0]):
                    lg, sc, _, _ = self.evaluator.eval_actions(
                        acts[j:j + 1],
                        filename=f"{self.logging_prefix}_oracle_{i}_{j}")
                    suc_all.append(float(np.asarray(sc, dtype=float).mean()))
                    d_all.append(float(lg.get("mean_vis_dist",
                                              lg.get("mean_state_dist", np.nan))))
                suc_all = np.array(suc_all)
                d_all = np.array(d_all)
                _orec["step"].append(i)
                _orec["task"].append(task.detach().cpu().numpy().copy())
                _orec["cost"].append(state.cost.detach().cpu().numpy().copy())
                _orec["V"].append(V_pop.detach().cpu().numpy().copy())
                _orec["success"].append(suc_all)
                _orec["d_true"].append(d_all)
                _orec["sel_idx"].append(best_idx)
                print(f"[EM-SMC] oracle step={i}: coverage={suc_all.max():.0f} "
                      f"selected={suc_all[best_idx]:.0f} "
                      f"n_success={int(suc_all.sum())}/{len(suc_all)} "
                      f"min_d={np.nanmin(d_all):.3f} d[sel]={d_all[best_idx]:.3f}")

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    best_action.detach(),
                    filename=f"{self.logging_prefix}_output_{i}")
                current_success = float(successes.mean())
                mean_dist = logs.get('mean_state_dist', np.inf)
                mean_vis_dist = logs.get('mean_visual_dist', np.inf)
                mean_proprio_dist = logs.get('mean_proprio_dist', np.inf)
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i,
                             f"{self.logging_prefix}/ess": ess,
                             f"{self.logging_prefix}/beta": beta,
                             f"{self.logging_prefix}/delta_EV": delta_EV,
                             f"{self.logging_prefix}/cum_kl": cum_kl,
                             f"{self.logging_prefix}/n_resamples": n_resamples,
                             f"{self.logging_prefix}/accept": acc,
                             f"{self.logging_prefix}/step_size": eps,
                             f"{self.logging_prefix}/loss": best_task,
                             f"{self.logging_prefix}/loss_best": best_ever_task,
                             f"{self.logging_prefix}/log_zhat": log_zhat,
                             f"{self.logging_prefix}/softmin_cost":
                                 softmin_cost(state.cost, target.alpha),
                             f"{self.logging_prefix}/softmin_cost_cold":
                                 softmin_cost(state.cost, 0.1 * target.alpha),
                             f"{self.logging_prefix}/cost_min":
                                 float(state.cost.min()),
                             f"{self.logging_prefix}/cost_wmean":
                                 float((w * state.cost).sum()),
                             f"{self.logging_prefix}/selected_mu": float(use_mu),
                             f"{self.logging_prefix}/rollout_seqs":
                                 target.n_rollout_seqs,
                             f"{self.logging_prefix}/V_selected":
                                 float(V_pop[best_idx]),
                             f"{self.logging_prefix}/V_pop_mean":
                                 float(V_pop.mean()),
                             f"{self.logging_prefix}/V_pop_std":
                                 float(V_pop.std())})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    if first_success_step is None:
                        first_success_step = i
                    if self.stop_on_success:
                        exit_step = i
                        break

        self.exit_step = (exit_step if first_success_step is None
                          else first_success_step)
        self.first_success_step = first_success_step
        self.total_measured_time = total_measured_time

        self.log_zhat = log_zhat
        self.softmin_cost = softmin_cost(state.cost, target.alpha)
        self.cost_min = float(state.cost.min())
        print(f"[EM-SMC] coverage: softminC={self.softmin_cost:.4f} "
              f"Cmin={self.cost_min:.4f} alpha={target.alpha:.4g} "
              f"| logZ*={log_zhat:+.3f} (target-motion diagnostic only, "
              f"biased by the adaptive mu -- not coverage)")
        print(f"[EM-SMC] done | beta={target.beta:.3f} | resamples={n_resamples} "
              f"| cumKL={cum_kl:.2f} nats | final ESS={ess_of(logw):.1f}/{P} "
              f"| {self.kernel} acc={acc:.3f} eps={eps:.4g} "
              f"| rollout seqs={target.n_rollout_seqs} "
              f"calls={target.n_rollout_calls}"
              + (" | WARNING: cumKL=0 -- the tilt never acted"
                 if cum_kl == 0.0 else ""))

        if _orec is not None and _orec["step"]:
            # `final_success`, not `success`: _orec already has a per-particle
            # "success" key and a duplicate keyword raises.
            np.savez(_oracle, **{k: np.array(v) for k, v in _orec.items()},
                     log_zhat=log_zhat, exit_step=exit_step,
                     final_success=current_success, kernel=self.kernel,
                     num_particles=P,
                     softmin_cost=softmin_cost(state.cost, target.alpha),
                     cost_min=float(state.cost.min()))
            print(f"[EM-SMC] oracle trace -> {_oracle} "
                  f"({len(_orec['step'])} checkpoints x {P} particles)")

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        fname = (f"seed_success_chunk_{task_id}.txt" if task_id
                 else "seed_success_log.txt")
        with open(os.path.join(os.environ.get("GRASP_RESULTS_DIR", os.getcwd()),
                              fname), "a") as f:
            # exit_step must report the FIRST success even when the run was
            # allowed to continue past it, or stop_on_success=False would look
            # like a failure to every downstream collator.
            f.write(f"{max(current_success, 1.0 if first_success_step is not None else 0.0)} "
                    f"{mean_dist} {mean_vis_dist} {mean_proprio_dist} "
                    f"{first_success_step if first_success_step is not None else exit_step}\n")

        # The run's argmin, not the last iteration's -- see the comment on
        # best_ever_action above.
        final_action = (best_ever_action if best_ever_action is not None
                        else best_action.detach())
        return final_action, np.full(1, np.inf)
