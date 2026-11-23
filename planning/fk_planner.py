"""AIS/SMC-sampler (Feynman-Kac family) planner: a GRASP-style Langevin
kernel over action-sequence particles, tilted toward low heat-solver
cost-to-go V by adaptive-tempered importance weights + systematic resampling.

The 2026-08-21 rewrite fixes three no-op bugs found in the first version (all
three together made the planner provably identical to vanilla GD):

  1. Degenerate population: all particles initialized to the same zero
     action sequence, so resampling selected among clones. Now particles
     start at init + N(0, init_std^2); particle 0 keeps the incumbent.
  2. Inert tilt: a fixed linear beta ramp with logw reset every
     resample_interval capped the per-window tilt exponent at
     beta_max*interval/opt_steps ~ 1/60 of V's std -- systematic resampling
     at that weight spread is the identity permutation. Now beta_t is
     ADAPTIVE: each step it advances to the largest value in [beta, beta_max]
     keeping ESS >= ess_target_frac * P (bisection), and resampling fires on
     the ESS threshold, not a timer.
  3. (Reverted 2026-08-21, same day: an intermediate version selected by
     argmin(L_dyn + beta*V). That is wrong -- it re-introduces V as an
     additive objective term, the mechanism already nulled in
     heat-value-equals-linear-bias, and the scales are incommensurable
     (L_dyn ~ 0.01-0.06 at convergence vs beta*V ~ 14), so it degenerated
     to argmin V. In the FK framework the tilt acts ONLY through sampling:
     resampling reshapes which particles exist, and selection is
     argmin L_dyn over that tilted population -- BoN on the loss over
     FK-tilted samples. With bugs 1-2 fixed, that selection is no longer
     vanilla GD's answer, because the candidate set itself is tilted.)

Weight bookkeeping uses the exact telescoped form: with a pi_0-invariant
mutation kernel (Adam on L_dyn + Langevin noise; V never in the gradient)
and targets pi_beta ~ pi_0 * exp(-beta*V), the accumulated SMC-sampler
log-weight between resamples telescopes to

    logw_i = beta_anchor * V_i(anchor) - beta_now * V_i(now),

where the anchor is the particle's state at the last resample. This both
implements the tempering increment -dbeta*V and the reversible-kernel
mutation correction -beta*dV, so the tilt keeps acting against kernel drift
even after beta_max is reached (the old version stopped tilting there).

Tilt observability (logged every step, wandb + stdout): ESS, realized beta
schedule, E_w[V]-E[V] (weighted-minus-uniform population mean of V; negative
means mass has moved toward low cost-to-go), per-resample selection-pressure
KL in nats with its cumulative sum, and unique-ancestor counts. cum_kl == 0
at the end of a run is the alarm that the tilt never acted (the exact
failure mode of the first version). cos(grad L_dyn, grad V) and dmin stay as
the redundancy/off-manifold diagnostics from the design memory.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

from .gd import GDPlanner


def systematic_resample(logw):
    """Systematic resampling (Douc & Cappe) from log-weights.

    Returns an index tensor of the same length as logw, sampled so that
    particle i survives with probability proportional to softmax(logw)[i].
    """
    w = torch.softmax(logw, dim=0)
    n = w.shape[0]
    positions = (torch.rand(1, device=w.device)
                 + torch.arange(n, device=w.device, dtype=torch.float32)) / n
    cumsum = torch.cumsum(w, dim=0)
    cumsum[-1] = 1.0  # guard against fp round-off leaving the last bin short
    idx = torch.searchsorted(cumsum, positions)
    return idx.clamp(max=n - 1)


def ess_of(logw):
    """Effective sample size 1/sum(w^2) of normalized weights."""
    w = torch.softmax(logw, dim=0)
    return 1.0 / (w * w).sum().item()


def selection_kl(logw):
    """KL(softmax(logw) || uniform) in nats: the selection pressure one
    resample applies to the population. 0 <=> resampling is the identity."""
    n = logw.shape[0]
    w = torch.softmax(logw, dim=0)
    return float((w * (torch.log(w.clamp_min(1e-30)) + math.log(n))).sum().item())


def adapt_beta(anchor, V, beta_lo, beta_max, ess_target, n_bisect=40):
    """Largest beta in [beta_lo, beta_max] with ESS(anchor - beta*V) >= ess_target.

    Standard adaptive tempering (Del Moral et al.): advance the tilt as fast
    as the population can pay for it. If even beta_max keeps ESS above
    target, returns beta_max (tilt is fully affordable). If ESS is already
    below target at beta_lo (mutation drift used up the budget), returns
    beta_lo unchanged -- the caller resamples.
    """
    if ess_of(anchor - beta_max * V) >= ess_target:
        return beta_max
    lo, hi = beta_lo, beta_max
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if ess_of(anchor - mid * V) >= ess_target:
            lo = mid
        else:
            hi = mid
    return lo


class FKSMCPlanner(GDPlanner):
    """Population-of-particles Langevin planner with adaptive AIS/SMC
    reweighting.

    Reuses GDPlanner's __init__ (replay buffer loading, heat-solver/graph
    compilation, _get_latent) unchanged; overrides plan() entirely. Plans
    one env at a time (matches WMRewardPlanner's existing convention of
    operating on obs_0[0]).
    """

    @torch.no_grad()
    def _dmin(self, z, chunk=4096):
        """Nearest-neighbor distance from z to the offline-data graph nodes,
        in the heat solver's scaled coordinates -- same primitive as
        WMRewardPlanner._lookup. Diagnostic only: rising dmin means the
        particles are drifting off the real-data manifold the v_surrogate
        was fit on.
        """
        zs = (z - self.heat_solver.proj_mean) / (self.heat_solver.proj_std + 1e-8)
        nodes = self.heat_solver.graph_nodes
        dmins = []
        for s in range(0, len(zs), chunk):
            d = torch.cdist(zs[s:s + chunk], nodes)
            dmins.append(d.min(dim=1).values)
        return torch.cat(dmins)

    def __init__(
        self,
        num_particles=32,
        num_langevin_steps=1,
        langevin_lr=0.05,
        langevin_temp=1e-3,
        beta_max=1.0,
        init_std=1.0,
        ess_target_frac=0.5,
        num_smooth_samples=8,
        smooth_sigma=1.0,
        anneal_frac=0.2,
        kernel="adam",
        v_null="none",
        flow_steps=10,
        flow_lr=0.05,
        flow_logdet_weight=1e-3,
        flow_shared=True,
        flow_scale=True,
        flow_weighted=True,
        flow_persist=False,
        flow_scale_lr=None,
        flow_type="diag",
        iaf_hids=30,
        iaf_layers=3,
        iaf_shift_only=False,
        iaf_shift_scale=1.0,
        iaf_bounded_shift=0.0,
        transport_first=False,
        iaf_per_particle=False,
        iaf_distill=False,
        prox_iters=3,
        jko_tau=0.0,
        jko_iters=4,
        hmc_eps=None,
        hmc_target_acc=0.65,
        hmc_leaps=1,
        hmc_beta1=0.9,
        hmc_warmup=10,
        svgd_bw=None,
        precond_noise=False,
        precond_freeze=10,
        precond_noise_scale=1.0,
        precond_normalize=False,
        freeze_tilt=True,
        **kwargs,
    ):
        kwargs.setdefault("sample_type", "zero")
        kwargs.setdefault("lr", langevin_lr)
        kwargs.setdefault("var_actions", 0.0)
        kwargs.setdefault("num_samples", 1)
        super().__init__(**kwargs)
        self.num_particles = num_particles
        self.num_langevin_steps = num_langevin_steps
        self.langevin_lr = langevin_lr
        self.langevin_temp = langevin_temp
        self.beta_max = beta_max
        self.init_std = init_std
        self.ess_target_frac = ess_target_frac
        self.num_smooth_samples = num_smooth_samples
        self.smooth_sigma = smooth_sigma
        self.anneal_frac = anneal_frac
        # Control knob for attributing FK's effect. "none" = real V.
        # "shuffle" permutes V across particles: identical marginal
        # distribution (so identical beta/ESS/resample-count scale), but the
        # particle->value association is destroyed, which is exactly the
        # information the tilt claims to use. "gauss" redraws from a matched
        # N(mean, std). Both leave the kernel untouched.
        # Mutation kernel. "adam" is the original and stays the default so every
        # existing result reproduces byte-for-byte.
        #   adam  a <- a - lr*Adam(grad L) + sqrt(2*lr*T)*xi
        #         The noise coefficient is already the correct Euler-Maruyama
        #         discretisation of Langevin for pi_0 ~ exp(-L/T); ONLY Adam's
        #         preconditioner breaks pi_0-invariance. That matters because the
        #         telescoped FK weight identity assumes an invariant kernel.
        #   sgld  same, with the preconditioner removed -> textbook ULA/SGLD.
        #         (SGLD rather than ULA because the gradient is already
        #         stochastic through the num_smooth_samples mollification.)
        #   mala  sgld proposal + a Metropolis accept/reject, so the kernel is
        #         exactly pi_0-invariant. Costs a SECOND rollout+backward per
        #         step for grad L at the proposal, and forces S=1: the MH ratio
        #         is invalid against a Monte-Carlo-noisy objective, and the
        #         mollified loss is exactly that. Dropping smoothing is not free
        #         -- it is the mechanism behind gd_reacher's 93%.
        #   aft   Annealed Flow Transport move (Arbel et al. 2021): a shared
        #         DiagonalAffine flow fit to the weighted loss replaces Adam.
        assert kernel in ("adam", "sgld", "mala", "aft", "jko", "hmc", "svgd"), kernel
        self.kernel = kernel
        self.flow_steps = flow_steps
        self.flow_lr = flow_lr
        # The -log|det dT| entropy term rewards expansion and nothing in the
        # loss opposes it along flat directions; unweighted it compounds the
        # scale step over step (population L reached 7e22 by step 82).
        self.flow_logdet_weight = flow_logdet_weight
        # Ablation switches between the adam kernel and AFT (defaults = AFT):
        #   flow_shared   one map for all particles (False: a map per particle)
        #   flow_scale    learn the diagonal scale s (False: shift only)
        #   flow_weighted weight the fit by the SMC weights (False: plain sum,
        #                 as the adam kernel's L_dyn.sum())
        self.flow_shared = flow_shared
        self.flow_scale = flow_scale
        self.flow_weighted = flow_weighted
        #   flow_persist  carry the Adam state (moments) across iterations; the
        #                 map itself restarts at identity each step. Reset on
        #                 resample, as the adam kernel resets its optimizer.
        self.flow_persist = flow_persist
        #   flow_scale_lr learning rate for the scale s (default: flow_lr). Adam
        #                 moves s by ~lr per step regardless of its gradient, and
        #                 the map is re-applied every iteration, so an lr=0.1
        #                 scale rescales every action dim ~10%/step -- multiplicative
        #                 noise that broke Point maze H10 seed 6 (shift-only solves).
        self.flow_scale_lr = flow_lr if flow_scale_lr is None else flow_scale_lr
        # flow_type "iaf": the reference repo's AffineInverseAutoregressiveFlow
        # (planning/aft_flows.py) -- ONE map shared by all particles whose shift
        # and scale depend on the input, re-initialised to the identity every
        # temperature step as in aft.py, fit on the weighted free energy
        #   sum_i w_i [L(T(a_i)) - c * log|det dT(a_i)|].
        # The flow_shared/flow_scale/flow_persist switches apply to "diag" only.
        assert flow_type in ("diag", "iaf", "resflow", "prox"), flow_type
        self.flow_type = flow_type
        self.iaf_hids = iaf_hids
        self.iaf_layers = iaf_layers
        self.iaf_shift_only = iaf_shift_only
        self.iaf_shift_scale = iaf_shift_scale
        self.iaf_bounded_shift = iaf_bounded_shift
        self.transport_first = transport_first
        # Ablation: one flow PER PARTICLE (not a pushforward, so not valid AFT --
        # it isolates 'shared map' from 'position-dependent displacement').
        self.iaf_per_particle = iaf_per_particle
        # Distillation (option 2): fit the flow to reproduce the per-particle Adam
        # step, || T(a_i) - (a_i + d_i) ||^2, with d_i the update the adam kernel
        # would apply. The transport is then numerically the gradient step, but
        # carried by a single shared map (a valid pushforward).
        self.iaf_distill = iaf_distill
        # prox: implicit (backward-Euler) step T(x) = x - lr * P * grad U(T(x)), solved
        # by prox_iters fixed-point iterations. No learned map: T is the INVERSE of the
        # i-ResNet layer y -> y + lr*P*grad U(y) (Behrmann et al. 2019), so it is a
        # normalizing flow, and it is the JKO/proximal step of the gradient flow. P is
        # Adam's diagonal preconditioner built from ENSEMBLE gradient statistics, so T
        # stays a function of x. log|det dT| = -log det(I + lr*P*Hess U) is not computed
        # (dropped, as elsewhere here) -- report it as dropped, not zero.
        self.prox_iters = prox_iters
        # JKO (kernel="jko"): each iteration solves, per particle,
        #     a_i <- argmin_a  L_dyn(a) + ||a - a_i||^2 / (2 tau)
        # approximately, with jko_iters Adam steps from a_i. This is the
        # Wasserstein-proximal (JKO) step for the potential energy E_pi[L_dyn];
        # with only a potential term the proximal problem separates across
        # particles, so no density/flow is needed. The entropy part of the full
        # JKO functional is NOT included (the Langevin noise stands in for it),
        # and the proximal solve is inexact (truncated inner loop).
        self.jko_tau = jko_tau
        self.jko_iters = jko_iters
        # kernel="hmc": preconditioned HMC. Mass matrix M = diag(sqrt(v_hat)) is built
        # from Adam's second moment and FROZEN after hmc_warmup iterations -- a mass
        # matrix that keeps adapting to the particles' own history would make the
        # kernel adaptive and NOT pi-invariant. With the MH accept below the kernel is
        # pi_0-invariant, which is what the telescoped FK weight identity assumes (the
        # adam kernel violates it). hmc_eps small => moves at the adam kernel's scale
        # and acceptance near 1; that is a valid kernel, but it does not by itself make
        # the planner better -- do not claim otherwise.
        self.hmc_eps = hmc_eps
        self.hmc_target_acc = hmc_target_acc
        self._hmc_eps_cur = None
        self.hmc_leaps = hmc_leaps
        self.hmc_beta1 = hmc_beta1
        self.hmc_warmup = hmc_warmup
        self._hmc_p = None
        self._hmc_M = None
        # kernel="svgd": the mutation is the kernel-embedded steepest-descent
        # direction of KL(q || pi) with pi ∝ exp(-L_dyn / T) (Pulido & van Leeuwen's
        # mapping particle filter; SVGD direction):
        #   v(a_i) = 1/P sum_j [ K(a_j,a_i) * (-grad L(a_j)/T) + grad_{a_j} K(a_j,a_i) ]
        # fed to the SAME Adam optimiser that FK uses. The KL itself is not a scalar
        # loss here (q's density is unknown) -- only its gradient is available.
        # NOTE: V stays out of this direction, so it is the steepest descent for the
        # L_dyn-only target, not for the V-tilted one.
        # bandwidth: median heuristic unless svgd_bw is set.
        self.svgd_bw = svgd_bw
        # precond_noise: scale the Langevin noise by the SAME preconditioner Adam
        # applies to the drift, i.e. sqrt(2*lr*T*P) with P = 1/(sqrt(v_hat)+eps),
        # frozen after precond_freeze iterations. With isotropic noise (the default)
        # the drift is preconditioned but the noise is not, so the chain does not
        # target exp(-L/T); with this flag the adam branch becomes pSGLD
        # (preconditioned ULA, Li et al. AAAI 2016).
        self.precond_noise = precond_noise
        self.precond_freeze = precond_freeze
        # freeze_tilt: once beta reaches beta_max the target sequence has ended
        # and resampling stops (see the comment at the resample gate). Setting
        # this False keeps resampling on the ESS threshold for the remaining
        # iterations, which is the ablation the appendix asks for -- it should
        # be WORSE, since the -beta*dV drift correction is exact only for a
        # pi0-invariant kernel and this one is a non-stationary optimizer.
        self.freeze_tilt = freeze_tilt
        # precond_noise_scale: multiplier on the pSGLD noise. P = 1/(sqrt(v_hat)+eps)
        # is NOT dimensionless -- where the ensemble gradient is small it is large, so
        # sqrt(P) can amplify the injected noise far past plain FK's sqrt(2*lr*T).
        # <1 shrinks the chain back toward FK (at the cost of exactness: the step then
        # targets exp(-L/(T*scale^2)), i.e. a colder chain, not a different stationary
        # form). Reported as the pSGLD temperature knob, not as a free parameter.
        self.precond_noise_scale = precond_noise_scale
        # precond_normalize: divide P by its mean before taking the square root, so
        # the TOTAL injected noise variance is exactly FK's sqrt(2*lr*T) budget and the
        # preconditioner only redistributes it across coordinates (more noise where the
        # ensemble gradient is flat, less where it is steep). Without this, sqrt(P)
        # carries its own units and pSGLD is both anisotropic AND hotter than FK, which
        # confounds the comparison: a pSGLD failure could be either effect. With it,
        # precond_noise_scale=1 differs from FK in shape alone, and P uniform recovers
        # FK exactly.
        self.precond_normalize = precond_normalize
        self._pn_P = None
        self._dm = self._dv = None
        self._dt = 0
        assert v_null in ("none", "shuffle", "gauss",
                          "objective", "objective_raw"), v_null
        self.v_null = v_null

    def _loss_and_grad(self, a, obs_batch, zg_batch, P, S, sigma_t, reg_weight):
        """(L_dyn detached (P,), grad wrt a (P,H,A)) for the sgld/mala kernels.

        Mirrors the adam branch's loss exactly: S mollification samples averaged,
        plus the annealed L2 action reg on the UNPERTURBED a.
        """
        a = a.detach().requires_grad_(True)
        if S > 1:
            eps = torch.randn(S, *a.shape, device=a.device) * sigma_t
            a_eval = (a.unsqueeze(0) + eps).flatten(0, 1)
            obs_eval = {k: repeat(v, "p ... -> (s p) ...", s=S) for k, v in obs_batch.items()} \
                if isinstance(obs_batch, dict) else repeat(obs_batch, "p ... -> (s p) ...", s=S)
            zg_eval = {k: repeat(v, "p ... -> (s p) ...", s=S) for k, v in zg_batch.items()}
        else:
            a_eval, obs_eval, zg_eval = a, obs_batch, zg_batch
        i_z, _ = self.wm.rollout(obs_0=obs_eval, act=a_eval)
        L_all = self.objective_fn(i_z, zg_eval, a_eval)
        L = L_all.view(S, P).mean(dim=0) if S > 1 else L_all
        if reg_weight > 0:
            L = L + reg_weight * (a ** 2).mean(dim=(1, 2))
        g = torch.autograd.grad(L.sum(), a)[0]
        return L.detach(), g.detach()

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(
            obs_0, obs_g, encode_to_vector=False)
        n_evals = trans_obs_0["visual"].shape[0] if isinstance(trans_obs_0, dict) else trans_obs_0.shape[0]
        assert n_evals == 1, "FKSMCPlanner plans one env at a time (matches WMRewardPlanner's convention)"

        init = self.init_actions(obs_0, actions).to(self.device)  # (1, horizon, action_dim)
        P = self.num_particles
        a = init.unsqueeze(0).repeat(P, 1, 1, 1).flatten(0, 1).clone().detach()
        # Population diversity at t=0: SMC over identical particles is
        # vanilla GD by construction (resampling among clones selects
        # nothing). Particle 0 keeps the incumbent for warm starts.
        a[1:] += torch.randn_like(a[1:]) * self.init_std
        a.requires_grad_(True)
        optimizer = torch.optim.Adam([a], lr=self.langevin_lr)

        # Fit the differentiable V surrogate for this goal, reusing the
        # existing offline-data heat solver -- no new distance/graph code.
        goal_latent = self._get_latent(z_obs_g)[0, 0].detach().cpu().numpy()
        self.heat_solver.solve_for_goal(goal_latent)

        if isinstance(trans_obs_0, dict):
            obs_batch = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=P) for k, v in trans_obs_0.items()}
        else:
            obs_batch = repeat(trans_obs_0[0].unsqueeze(0), "1 ... -> n ...", n=P)
        zg_batch = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=P).detach() for k, v in z_obs_g.items()}

        # Offline prior scan (GRASP_PRIOR_SCAN=<path.npz>): draw N action
        # sequences from the SAME prior the particles come from --
        # N(z, init_std^2) with z = init_actions() -- and evaluate V and the
        # task loss on each under no_grad. This is the density the FK tilt
        # reweights by exp(-beta*V); 3 particles are just a sparse MC estimate
        # of it, so scanning it directly is what makes the tilt visualisable.
        _scan = os.environ.get("GRASP_PRIOR_SCAN")
        if _scan:
            N = int(os.environ.get("GRASP_PRIOR_N", "512"))
            B = int(os.environ.get("GRASP_PRIOR_B", "8"))
            base = init[0].unsqueeze(0).repeat(N, 1, 1)
            noise = torch.randn_like(base) * self.init_std
            noise[0] = 0.0                      # keep the prior mean as sample 0
            samples = (base + noise).detach()
            Vs, Ls = [], []
            for st in range(0, N, B):
                ab = samples[st:st + B]
                n = ab.shape[0]
                ob = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=n)
                      for k, v in trans_obs_0.items()} if isinstance(trans_obs_0, dict)                     else repeat(trans_obs_0[0].unsqueeze(0), "1 ... -> n ...", n=n)
                zg = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=n).detach()
                      for k, v in z_obs_g.items()}
                with torch.no_grad():
                    i_z, _ = self.wm.rollout(obs_0=ob, act=ab)
                    zT = self._get_latent(i_z)[:, -1, :]
                    Vs.append(self.heat_solver.evaluate_rollout(zT).detach().cpu().numpy())
                    Ls.append(self.objective_fn(i_z, zg, ab).detach().cpu().numpy())
                if st % (B * 16) == 0:
                    print(f"[prior-scan] {st + n}/{N}")
            np.savez(_scan, a=samples.cpu().numpy(),
                     V=np.concatenate(Vs), L=np.concatenate(Ls),
                     z0=init[0].detach().cpu().numpy(), init_std=self.init_std)
            print(f"[prior-scan] saved {N} samples -> {_scan}")
            return init.repeat(n_evals, 1, 1), np.full(n_evals, np.inf)

        # Plane scan (GRASP_PLANE_SCAN=<in.npz>, GRASP_PLANE_OUT=<out.npz>):
        # evaluate V and the task loss on a 2-D GRID over the affine plane
        # z0 + x*u + y*v, where u,v are supplied (typically prior-mean ->
        # tilt solution and prior-mean -> no-tilt solution, orthogonalised).
        # The prior is isotropic in 100-D, so PCA gives arbitrary axes; a plane
        # through the two actual outcomes is the informative slice, and a grid
        # gives a true landscape instead of a scatter projection.
        _plane = os.environ.get("GRASP_PLANE_SCAN")
        if _plane:
            pin = np.load(_plane)
            u = torch.tensor(pin["u"], device=self.device, dtype=init.dtype)
            v = torch.tensor(pin["v"], device=self.device, dtype=init.dtype)
            z0 = torch.tensor(pin["z0"], device=self.device, dtype=init.dtype)
            xs = torch.tensor(pin["xs"], device=self.device, dtype=init.dtype)
            ys = torch.tensor(pin["ys"], device=self.device, dtype=init.dtype)
            B = int(os.environ.get("GRASP_PRIOR_B", "8"))
            grid = torch.stack([z0 + x * u + y * v for y in ys for x in xs])
            Vs, Ls = [], []
            for st in range(0, grid.shape[0], B):
                ab = grid[st:st + B]; n = ab.shape[0]
                ob = {k: repeat(v2[0].unsqueeze(0), "1 ... -> n ...", n=n)
                      for k, v2 in trans_obs_0.items()}
                zg = {k: repeat(v2[0].unsqueeze(0), "1 ... -> n ...", n=n).detach()
                      for k, v2 in z_obs_g.items()}
                with torch.no_grad():
                    i_z, _ = self.wm.rollout(obs_0=ob, act=ab)
                    zT = self._get_latent(i_z)[:, -1, :]
                    Vs.append(self.heat_solver.evaluate_rollout(zT).detach().cpu().numpy())
                    Ls.append(self.objective_fn(i_z, zg, ab).detach().cpu().numpy())
                if st % (B * 20) == 0:
                    print(f"[plane-scan] {st + n}/{grid.shape[0]}")
            out = os.environ.get("GRASP_PLANE_OUT", _plane.replace(".npz", "_out.npz"))
            np.savez(out, V=np.concatenate(Vs).reshape(len(ys), len(xs)),
                     L=np.concatenate(Ls).reshape(len(ys), len(xs)),
                     xs=pin["xs"], ys=pin["ys"])
            print(f"[plane-scan] saved {grid.shape[0]} grid points -> {out}")
            return init.repeat(n_evals, 1, 1), np.full(n_evals, np.inf)

        # Telescoped SMC-sampler weights: logw = anchor - beta*V(now), with
        # anchor_i = beta_at_last_resample * V_i(at last resample). Covers
        # both the tempering increment and the reversible-kernel mutation
        # correction (see module docstring). beta starts at 0 <=> logw = -b*V.
        anchor = torch.zeros(P, device=self.device)
        beta = 0.0
        self._flow_state = None
        self._prev_w = torch.full((P,), 1.0 / P, device=self.device)
        self._moved_this_step = False
        ess_target = self.ess_target_frac * P

        # Optional per-step dump for visualising the tilted action
        # distribution (GRASP_FK_DUMP=<path.npz>): every particle's action, its
        # V, its loss and the resampling ancestry, so the tilt can be
        # reconstructed offline without re-running the world model.
        _dump_path_fk = os.environ.get("GRASP_FK_DUMP")
        _fk_rec = {"a": [], "V": [], "L": [], "beta": [], "w": [],
                   "idx": [], "resampled": [], "step": [], "V_roadmap": []} \
            if _dump_path_fk else None

        n_resamples, cum_kl = 0, 0.0
        last_unique = P
        total_measured_time = 0.0
        exit_step = self.opt_steps
        current_success = 0.0
        mean_dist = mean_vis_dist = mean_proprio_dist = np.inf
        L_dyn = None
        V_diag = None

        for i in range(self.opt_steps):
            start = time.perf_counter()
            # Alg. 1 order (transport_first): move the particles BEFORE the
            # weights are computed, so V is read at the transported points.
            self._moved_this_step = False
            if self.transport_first and self.kernel == "aft" and self.flow_type in ("iaf", "resflow"):
                sig0 = self.smooth_sigma * max(0.0, 1.0 - (i / float(self.opt_steps)))
                S0 = self.num_smooth_samples if sig0 > 0 else 1
                reg0 = self.action_reg * max(0.0, 1.0 - (i / float(self.opt_steps))) \
                    if self.action_reg > 0 else 0.0
                self._pending_L, _ = self._aft_iaf_move(
                    a, self._prev_w, obs_batch, zg_batch, P, S0, sig0, reg0)
                self._moved_this_step = True

            # --- V at current particles: forward-eval only, no grad ---
            with torch.no_grad():
                i_z_obses, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                z_T = self._get_latent(i_z_obses)[:, -1, :]
                V_val = self.heat_solver.evaluate_rollout(z_T).detach()
                _V_roadmap = V_val.clone()      # pre-substitution, for the dump
                if self.v_null == "shuffle":
                    V_val = V_val[torch.randperm(V_val.numel(), device=V_val.device)]
                elif self.v_null == "gauss":
                    V_val = V_val.mean() + V_val.std() * torch.randn_like(V_val)
                elif self.v_null in ("objective", "objective_raw"):
                    # Heat-equation ablation: tilt on the SAME L2 objective the
                    # kernel descends, instead of the roadmap cost-to-go. If the
                    # heat solve contributes nothing beyond the objective, these
                    # should match the real runs.
                    #
                    # 'objective_raw' substitutes L_dyn AS IT IS, and that is
                    # the ablation the paper's claim calls for: the claim is that
                    # a PDE-derived cost is better suited NUMERICALLY for
                    # tilting, so its spread over the population (V on [0,20] vs
                    # L_dyn ~1-2.6) is part of what is under test, not a
                    # confound to be removed.
                    # 'objective' rescales L_dyn to V's per-step mean/std. That
                    # hands L2 the very property being tested and reduces the
                    # comparison to one of ORDERING only -- use it to isolate
                    # ordering, never as the headline ablation.
                    L_tilt = self.objective_fn(i_z_obses, zg_batch, a).detach()
                    if self.v_null == "objective":
                        L_tilt = (L_tilt - L_tilt.mean()) / (L_tilt.std() + 1e-8)
                        L_tilt = V_val.mean() + V_val.std() * L_tilt
                    V_val = L_tilt

            # --- adaptive tempering: advance beta as far as ESS affords ---
            beta_prev = beta
            anneal_done = beta >= self.beta_max
            if not anneal_done:
                beta = adapt_beta(anchor, V_val, beta, self.beta_max, ess_target)
                # Schedule floor: the anneal must complete by
                # anneal_frac*opt_steps. On goals where V's population spread
                # makes ESS-adaptive beta creep (seed 9: 49 resamples, anneal
                # done only at step 48, kernel then converged monotonically
                # but ran out of budget), the churn eats the optimization
                # phase. The floor caps total selection pressure and
                # guarantees a long clean kernel phase after the freeze.
                anneal_steps = max(1, int(self.anneal_frac * self.opt_steps))
                beta = max(beta, self.beta_max * min(1.0, i / anneal_steps))
            logw = anchor - beta * V_val
            ess = ess_of(logw)
            w = torch.softmax(logw, dim=0)
            self._prev_w = w.detach()
            # Instantaneous tilt effect: weighted-minus-uniform mean of V.
            delta_EV = float((w * V_val).sum().item() - V_val.mean().item())

            if _fk_rec is not None:
                _fk_rec["step"].append(i)
                _fk_rec["a"].append(a.detach().cpu().numpy().copy())
                _fk_rec["V"].append(V_val.detach().cpu().numpy().copy())
                _fk_rec["V_roadmap"].append(_V_roadmap.detach().cpu().numpy().copy())
                _fk_rec["beta"].append(float(beta))
                _fk_rec["w"].append(w.detach().cpu().numpy().copy())

            # --- resample on ESS threshold (not a timer), ONLY while the
            # anneal is running. A budget-limited beta advance leaves ESS
            # pinned just above target (bisection approaches from below), so
            # it must also trigger a resample or the anneal stalls. Once
            # beta hits beta_max the target sequence has ended: the -beta*dV
            # drift correction is exact only for a pi0-invariant kernel, and
            # this kernel is an optimizer (non-stationary, always descending
            # L), so post-anneal reweighting fights the L-descent it needs --
            # observed as 145-resample churn that broke seed 9 while the
            # anneal-phase tilt alone accelerated seed 10 by 7x. Tilt =
            # annealed prior over basins; the kernel finishes the job. ---
            # The completing jump to beta_max also resamples once (realizing
            # the final tilt window) before the freeze.
            budget_limited_advance = beta > beta_prev and beta < self.beta_max
            completing_jump = (not anneal_done) and beta >= self.beta_max
            tilt_live = (not anneal_done) or (not self.freeze_tilt)
            if tilt_live and (ess < ess_target or budget_limited_advance or completing_jump):
                kl = selection_kl(logw)
                with torch.no_grad():
                    idx = systematic_resample(logw)
                    a.data = a.data[idx].clone()
                    anchor = beta * V_val[idx]
                    w = torch.full_like(w, 1.0 / P)   # post-resample weights (aft fit)
                if _fk_rec is not None:
                    _fk_rec["idx"].append(idx.detach().cpu().numpy().copy())
                    _fk_rec["resampled"].append(i)
                cum_kl += kl
                n_resamples += 1
                last_unique = int(idx.unique().numel())
                # Adam's per-particle state is tied to tensor position;
                # resampling reshuffles particle identity, so reset it.
                optimizer = torch.optim.Adam([a], lr=self.langevin_lr)
                if not self.iaf_distill:
                    self._flow_state = None

            # --- Mutation kernel: gd_reacher's PROVEN smoothed-annealed
            # gradient step (gd.py:993-1034), applied per particle. Each
            # particle's loss is averaged over num_smooth_samples Gaussian
            # perturbations with annealed sigma (Gaussian-convolved loss --
            # GRASP Thm 4 landscape smoothing, coarse-to-fine), plus the
            # annealed L2 action reg (gd.py's anti-exploitation device).
            # The FK corrector (V, beta) never enters the loss/gradient. ---
            cos_sim_val = 0.0
            dmin_val = 0.0
            sigma_t = self.smooth_sigma * max(0.0, 1.0 - (i / float(self.opt_steps)))
            S = self.num_smooth_samples if sigma_t > 0 else 1
            reg_w = self.action_reg * max(0.0, 1.0 - (i / float(self.opt_steps))) \
                if self.action_reg > 0 else 0.0

            if self.kernel in ("sgld", "mala"):
                # --- pi_0-invariant (or nearly so) alternatives to the Adam
                # branch below. Same noise coefficient sqrt(2*lr*T): that was
                # already the correct Langevin discretisation; what changes is
                # that the drift is the RAW gradient, not a preconditioned one.
                lr, T = self.langevin_lr, self.langevin_temp
                S_k = 1 if self.kernel == "mala" else S   # MH needs a noiseless L
                for _ in range(self.num_langevin_steps):
                    L_cur, g_cur = self._loss_and_grad(a, obs_batch, zg_batch, P, S_k, sigma_t, reg_w)
                    xi = torch.randn_like(a) * math.sqrt(2 * lr * T)
                    a_prop = a.detach() - lr * g_cur + xi
                    if self.kernel == "sgld":
                        a.data = a_prop
                        accept_rate = 1.0
                    else:
                        L_prop, g_prop = self._loss_and_grad(a_prop, obs_batch, zg_batch, P, S_k, sigma_t, reg_w)
                        # log q(x'|x) = -||x' - (x - lr*grad L(x))||^2 / (4*lr*T)
                        sq = lambda z: (z ** 2).flatten(1).sum(1)
                        log_q_fwd = -sq(a_prop - (a.detach() - lr * g_cur)) / (4 * lr * T)
                        log_q_rev = -sq(a.detach() - (a_prop - lr * g_prop)) / (4 * lr * T)
                        log_alpha = (L_cur - L_prop) / T + log_q_rev - log_q_fwd
                        acc = torch.rand(P, device=a.device) < torch.exp(log_alpha.clamp(max=0.0))
                        a.data = torch.where(acc.view(-1, *([1] * (a.dim() - 1))), a_prop, a.data)
                        accept_rate = float(acc.float().mean().item())
                        L_cur = torch.where(acc, L_prop, L_cur)
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)
                L_dyn = L_cur
                with torch.no_grad():
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                self._last_accept = accept_rate
                _skip_adam = True
            elif self.kernel == "svgd":
                T_ = max(self.langevin_temp, 1e-12)
                L_cur, g = self._loss_and_grad(a, obs_batch, zg_batch, P, S, sigma_t, reg_w)
                with torch.no_grad():
                    x = a.detach().reshape(P, -1)
                    d2 = torch.cdist(x, x) ** 2
                    if self.svgd_bw:
                        h = float(self.svgd_bw)
                    else:                                  # median heuristic
                        med = d2[d2 > 0].median() if (d2 > 0).any() else torch.tensor(1.0, device=x.device)
                        h = float(med.item()) / max(math.log(P + 1.0), 1e-8)
                        h = max(h, 1e-8)
                    K = torch.exp(-d2 / h)                 # (P,P) RBF
                    score = (-g / T_).reshape(P, -1)       # grad log pi at each particle
                    attract = K @ score                    # sum_j K(a_j,a_i) score(a_j)
                    # sum_j grad_{a_j} K(a_j,a_i) for the RBF kernel
                    repel = (2.0 / h) * (K.sum(dim=1, keepdim=True) * x - K @ x)
                    v = (attract + repel) / P
                    a.grad = (-v).reshape_as(a)            # Adam ascends -v = descends KL
                optimizer.step()
                with torch.no_grad():
                    if self.langevin_temp > 0:
                        a.data += torch.randn_like(a) * math.sqrt(2 * self.langevin_lr * self.langevin_temp)
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)
                    L_dyn = L_cur.detach()
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                    self._last_flow = (float(repel.abs().mean().item()), float(attract.abs().mean().item()))
                _skip_adam = True
            elif self.kernel == "hmc":
                T_ = max(self.langevin_temp, 1e-12)
                b2 = 0.999
                # --- mass matrix from the second moment of grad U = grad L / T (NOT of
                # grad L: the metric must match the energy, or dH explodes and every
                # proposal is rejected -- measured acceptance 0.00 before this fix) ---
                L_cur, g = self._loss_and_grad(a, obs_batch, zg_batch, P, S, sigma_t, reg_w)
                gU = g / T_
                if self._hmc_M is None:
                    if self._dv is None:
                        self._dv = torch.zeros_like(gU); self._dt = 0
                    self._dt += 1
                    self._dv = b2 * self._dv + (1 - b2) * gU * gU
                    if i >= self.hmc_warmup:
                        self._hmc_M = (self._dv / (1 - b2 ** self._dt)).sqrt() + 1e-8
                M = self._hmc_M if self._hmc_M is not None else \
                    ((self._dv / (1 - b2 ** self._dt)).sqrt() + 1e-8)
                # --- step size: adapted to hmc_target_acc during warm-up, then frozen
                # (a step size that kept adapting would break pi-invariance) ---
                if self._hmc_eps_cur is None:
                    self._hmc_eps_cur = self.hmc_eps if self.hmc_eps else 0.1
                eps = self._hmc_eps_cur
                # --- momentum: partial refresh (hmc_beta1=0 is full refresh) ---
                if self._hmc_p is None or self._hmc_p.shape != a.shape:
                    self._hmc_p = torch.zeros_like(a)
                b1 = self.hmc_beta1
                self._hmc_p = b1 * self._hmc_p + math.sqrt(max(0.0, 1 - b1 ** 2)) * torch.randn_like(a) * M.sqrt()
                p0, a0, U0 = self._hmc_p.clone(), a.detach().clone(), L_cur / T_
                K0 = 0.5 * (p0 * p0 / M).flatten(1).sum(1)
                # --- leapfrog on U = L_dyn / T with metric M ---
                pcur, acur, gcur = p0.clone(), a0.clone(), gU.clone()
                for _ in range(max(1, self.hmc_leaps)):
                    pcur = pcur - 0.5 * eps * gcur
                    acur = acur + eps * pcur / M
                    L_new, g_new = self._loss_and_grad(acur, obs_batch, zg_batch, P, S, sigma_t, reg_w)
                    gcur = g_new / T_
                    pcur = pcur - 0.5 * eps * gcur
                U1 = L_new / T_
                K1 = 0.5 * (pcur * pcur / M).flatten(1).sum(1)
                log_alpha = (U0 + K0) - (U1 + K1)
                acc = torch.rand(P, device=a.device) < torch.exp(log_alpha.clamp(max=0.0))
                with torch.no_grad():
                    m_ = acc.view(-1, *([1] * (a.dim() - 1)))
                    a.data = torch.where(m_, acur, a0)
                    self._hmc_p = torch.where(m_, pcur, -p0)      # flip on reject
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)
                    L_dyn = torch.where(acc, L_new, L_cur).detach()
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                self._last_accept = float(acc.float().mean().item())
                if i < self.hmc_warmup and self.hmc_eps is None:       # dual-averaging-style adaptation
                    self._hmc_eps_cur *= math.exp(0.5 * (self._last_accept - self.hmc_target_acc))
                self._last_flow = (self._last_accept, (a.data - a0).abs().mean().item())
                _skip_adam = True
            elif self.kernel == "jko":
                tau = self.jko_tau if self.jko_tau > 0 else self.langevin_lr
                a_prev = a.detach().clone()
                jopt = torch.optim.Adam([a], lr=self.langevin_lr)
                for _ in range(max(1, self.jko_iters)):
                    jopt.zero_grad()
                    L_j = self._smoothed_loss(a, obs_batch, zg_batch, P, S, sigma_t)
                    if reg_w > 0:
                        L_j = L_j + reg_w * (a ** 2).mean(dim=(1, 2))
                    prox = ((a - a_prev) ** 2).sum(dim=(1, 2)) / (2 * tau)
                    (L_j + prox).sum().backward()
                    jopt.step()
                with torch.no_grad():
                    if self.langevin_temp > 0:
                        a.data += torch.randn_like(a) * math.sqrt(2 * self.langevin_lr * self.langevin_temp)
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)
                    L_dyn = L_j.detach()
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                    self._last_flow = (0.0, (a.data - a_prev).abs().mean().item())
                _skip_adam = True
            elif self.kernel == "aft" and self.flow_type == "prox":
                b1, b2, eps = 0.9, 0.999, 1e-8
                x0 = a.detach().clone()
                y = x0.clone()
                # Adam's moments are coefficients OF THE MAP, carried across iterations
                # (as AFT's flow parameters are) and shared across particles so that T
                # stays a function of x. Momentum matters: without it the step is a
                # plain preconditioned gradient and cube seed 24 never crossed the
                # threshold (FK gets there at step 2 with momentum).
                if self._dm is None:
                    self._dm = torch.zeros_like(x0[:1]); self._dv = torch.zeros_like(x0[:1]); self._dt = 0
                m_prev, v_prev, t_prev = self._dm, self._dv, self._dt
                for _ in range(max(1, self.prox_iters)):     # fixed point: y <- x0 - lr*P*grad U(y)
                    L_y, g = self._loss_and_grad(y, obs_batch, zg_batch, P, S, sigma_t, reg_w)
                    g_ens = g.mean(dim=0, keepdim=True)
                    m = b1 * m_prev + (1 - b1) * g_ens
                    v = b2 * v_prev + (1 - b2) * (g * g).mean(dim=0, keepdim=True)
                    t = t_prev + 1
                    mhat, vhat = m / (1 - b1 ** t), v / (1 - b2 ** t)
                    # per-particle direction, ensemble preconditioner + ensemble momentum
                    step = (g + (mhat - g_ens)) / (vhat.sqrt() + eps)
                    y = x0 - self.langevin_lr * step
                self._dm, self._dv, self._dt = m, v, t
                with torch.no_grad():
                    a.data = y
                    if self.langevin_temp > 0:
                        a.data += torch.randn_like(a) * math.sqrt(2 * self.langevin_lr * self.langevin_temp)
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)
                    L_dyn = L_y.detach()
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                    self._last_flow = (0.0, (a.data - x0).abs().mean().item())
                _skip_adam = True
            elif self.kernel == "aft" and self.flow_type in ("iaf", "resflow"):
                if self._moved_this_step:          # transport already done at the top
                    L_dyn = self._pending_L
                else:
                    L_dyn, _ = self._aft_iaf_move(a, w, obs_batch, zg_batch, P, S, sigma_t, reg_w)
                with torch.no_grad():
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                _skip_adam = True
            elif self.kernel == "aft":
                # --- Annealed-flow-transport move (Arbel et al. 2021): instead
                # of per-particle Adam steps, fit ONE DiagonalAffine map
                # T(a) = a*exp(s) + b, shared by all particles, by minimising the
                # weighted KL free energy of the SAME loss the adam branch
                # descends (smoothed L_dyn + annealed action reg, unscaled):
                #     sum_i w_i L_dyn(T(a_i)) - c*log|det dT|,  log|det dT| = sum(s),
                # with c = flow_logdet_weight << 1 (see __init__)
                # (the -U_{k-1}(a) term does not depend on T and is dropped).
                # Then transport every particle through T. V/beta are untouched:
                # they act only through the weights above, exactly as before.
                # Fresh identity init each step unless flow_persist; Langevin noise
                # as in the adam kernel; no MCMC.
                if self.flow_persist and self._flow_state is not None:
                    # Keep Adam's moments, NOT the map: T_k is a fresh increment
                    # applied to the current particles. Persisting (s, b) would
                    # re-apply the old shift every iteration (b acts as an
                    # undamped velocity -- L4 diverged to qpos err 2.4 that way).
                    s_, b_, fopt = self._flow_state
                    with torch.no_grad():
                        s_.zero_()
                        b_.zero_()
                else:
                    pshape = a.shape[1:] if self.flow_shared else a.shape
                    s_ = torch.zeros(pshape, device=a.device, requires_grad=self.flow_scale)
                    b_ = torch.zeros(pshape, device=a.device, requires_grad=True)
                    fopt = torch.optim.Adam(
                        [{"params": [b_], "lr": self.flow_lr}]
                        + ([{"params": [s_], "lr": self.flow_scale_lr}] if self.flow_scale else []))
                w_fit = w.detach() if self.flow_weighted else torch.ones_like(w)
                x0 = a.detach()
                for _ in range(self.flow_steps):
                    y = x0 * torch.exp(s_) + b_
                    L_y = self._smoothed_loss(y, obs_batch, zg_batch, P, S, sigma_t)
                    if reg_w > 0:
                        L_y = L_y + reg_w * (y ** 2).mean(dim=(1, 2))
                    free_energy = (w_fit * L_y).sum() - self.flow_logdet_weight * s_.sum()
                    fopt.zero_grad()
                    free_energy.backward()
                    fopt.step()
                with torch.no_grad():
                    a.data = x0 * torch.exp(s_) + b_
                    # Same Langevin noise as the adam kernel (langevin_temp=0 disables).
                    if self.langevin_temp > 0:
                        a.data += torch.randn_like(a) * math.sqrt(2 * self.langevin_lr * self.langevin_temp)
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)
                    L_dyn = L_y.detach()
                    z_ng, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                    V_diag = self.heat_solver.evaluate_rollout(self._get_latent(z_ng)[:, -1, :])
                    dmin_val = self._dmin(z_T.detach()).mean().item()
                self._last_flow = (s_.abs().mean().item(), b_.abs().mean().item())
                if self.flow_persist:
                    self._flow_state = (s_, b_, fopt)
                _skip_adam = True
            else:
                _skip_adam = False

            for _ in range(0 if _skip_adam else self.num_langevin_steps):
                optimizer.zero_grad()
                if S > 1:
                    eps = torch.randn(S, *a.shape, device=a.device) * sigma_t
                    a_eval = (a.unsqueeze(0) + eps).flatten(0, 1)          # s-major (S*P)
                    obs_eval = {k: repeat(v, "p ... -> (s p) ...", s=S) for k, v in obs_batch.items()} \
                        if isinstance(obs_batch, dict) else repeat(obs_batch, "p ... -> (s p) ...", s=S)
                    zg_eval = {k: repeat(v, "p ... -> (s p) ...", s=S) for k, v in zg_batch.items()}
                else:
                    a_eval, obs_eval, zg_eval = a, obs_batch, zg_batch
                i_z_obses, _ = self.wm.rollout(obs_0=obs_eval, act=a_eval)
                L_all = self.objective_fn(i_z_obses, zg_eval, a_eval)
                L_dyn = L_all.view(S, P).mean(dim=0) if S > 1 else L_all
                if self.action_reg > 0:
                    reg_weight = self.action_reg * max(0.0, 1.0 - (i / float(self.opt_steps)))
                    L_dyn = L_dyn + reg_weight * (a ** 2).mean(dim=(1, 2))
                z_T_all = self._get_latent(i_z_obses)[:, -1, :]
                V_all = self.heat_solver.evaluate_rollout(z_T_all)
                V_diag = V_all.view(S, P).mean(dim=0) if S > 1 else V_all

                # grad_V here is diagnostic only (redundancy check) -- V
                # never contributes to the actual step.
                grad_L = torch.autograd.grad(L_dyn.sum(), a, retain_graph=True)[0]
                grad_V = torch.autograd.grad(V_diag.sum(), a, retain_graph=True)[0]
                with torch.no_grad():
                    cos_sim_val = F.cosine_similarity(
                        grad_L.flatten(1), grad_V.flatten(1), dim=1
                    ).mean().item()
                    dmin_val = self._dmin(z_T.detach()).mean().item()  # clean z_T from the no-grad block

                L_dyn.sum().backward()
                optimizer.step()

                with torch.no_grad():
                    scale = math.sqrt(2 * self.langevin_lr * self.langevin_temp)
                    if self.precond_noise:
                        st = optimizer.state.get(a, {})
                        if "exp_avg_sq" in st:
                            bc = 1 - 0.999 ** max(1, int(st.get("step", torch.tensor(1)).item()
                                                         if torch.is_tensor(st.get("step", 1))
                                                         else st.get("step", 1)))
                            Pm = 1.0 / ((st["exp_avg_sq"] / bc).sqrt() + 1e-8)
                            if self.precond_normalize:
                                Pm = Pm / Pm.mean().clamp_min(1e-12)
                            if self._pn_P is None or i < self.precond_freeze:
                                self._pn_P = Pm            # freeze after warm-up
                                if i == self.precond_freeze - 1:
                                    print(f"[psgld] mean sqrt(P) at freeze: "
                                          f"{self._pn_P.sqrt().mean().item():.4g}")
                        if self._pn_P is not None:
                            noise = (torch.randn_like(a) * scale
                                     * self.precond_noise_scale * self._pn_P.sqrt())
                        else:
                            noise = torch.randn_like(a) * scale
                    else:
                        noise = torch.randn_like(a) * scale
                    a.data += noise
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)

            total_measured_time += time.perf_counter() - start

            # Selection is argmin L_dyn over the FK-tilted population: the
            # tilt acts only through sampling (resampling reshaped which
            # particles exist), never as an objective term. V_sel-vs-pop
            # shows whether the loss winner is also a low-V particle.
            best_idx = int(torch.argmin(L_dyn.detach()))
            if _fk_rec is not None:
                _fk_rec["L"].append(L_dyn.detach().cpu().numpy().copy())

            if i % self.eval_every == 0:
                print(f"[FK-SMC] step={i:4d} | beta={beta:.3f} | L_dyn={L_dyn[best_idx].item():.4f} "
                      f"| V_sel={V_diag[best_idx].item():.4f} (pop {V_val.mean().item():.2f}) "
                      f"| ESS={ess:.1f}/{P} "
                      f"| dEV={delta_EV:+.3f} | resamples={n_resamples} (uniq={last_unique}) "
                      f"| cumKL={cum_kl:.2f} "
                      f"| cos_sim={cos_sim_val:.4f} | dmin={dmin_val:.4f}"
                      + (f" | kernel={self.kernel} acc={getattr(self, '_last_accept', 1.0):.3f}"
                         if self.kernel in ("sgld", "mala") else "")
                      + ((" | kernel=aft "
                          + (f"logdet={self._last_flow[0]:.4f} disp={self._last_flow[1]:.4f} "
                             + (f"derr={getattr(self, '_distill_err', float('nan')):.4f}/{getattr(self, '_distill_tgt', float('nan')):.4f} "
                                if self.iaf_distill else "")
                             if self.kernel in ("jko", "hmc", "svgd") or self.flow_type in ("iaf", "resflow", "prox") else
                             f"|s|={self._last_flow[0]:.4f} |b|={self._last_flow[1]:.4f} ")
                          + f"| L_pop={L_dyn.mean().item():.3g}")
                         if self.kernel in ("aft", "jko", "hmc", "svgd") else ""))

            if self.evaluator is not None and i % self.eval_every == 0:
                best_action = a[best_idx:best_idx + 1].detach()
                logs, successes, _, _ = self.evaluator.eval_actions(
                    best_action, filename=f"{self.logging_prefix}_output_{i}")
                current_success = float(successes.mean())
                mean_dist = logs.get('mean_state_dist', np.inf)
                mean_vis_dist = logs.get('mean_visual_dist', np.inf)
                mean_proprio_dist = logs.get('mean_proprio_dist', np.inf)
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i,
                             f"{self.logging_prefix}/cos_sim": cos_sim_val,
                             f"{self.logging_prefix}/ess": ess,
                             f"{self.logging_prefix}/beta": beta,
                             f"{self.logging_prefix}/delta_EV": delta_EV,
                             f"{self.logging_prefix}/cum_kl": cum_kl,
                             f"{self.logging_prefix}/n_resamples": n_resamples,
                             f"{self.logging_prefix}/V_selected": float(V_diag[best_idx].item()),
                             f"{self.logging_prefix}/V_pop_mean": float(V_val.mean().item()),
                             f"{self.logging_prefix}/V_pop_std": float(V_val.std().item())})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    exit_step = i
                    break

        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        print(f"[FK-SMC] done | beta={beta:.3f} | resamples={n_resamples} | "
              f"cumKL={cum_kl:.2f} nats | final ESS={ess:.1f}/{P}"
              + (" | WARNING: cumKL=0 -- the tilt never acted" if cum_kl == 0.0 else ""))

        if _fk_rec is not None:
            # One planning call per environment step, so a fixed path would keep
            # only the last call. Number them: <stem>_call<k>.npz.
            self._fk_dump_call = getattr(self, "_fk_dump_call", -1) + 1
            _stem, _ext = os.path.splitext(_dump_path_fk)
            _dump_path_fk = f"{_stem}_call{self._fk_dump_call:03d}{_ext or '.npz'}"
            np.savez(_dump_path_fk,
                     a=np.array(_fk_rec["a"]), V=np.array(_fk_rec["V"]),
                     L=np.array(_fk_rec["L"]), beta=np.array(_fk_rec["beta"]),
                     V_roadmap=np.array(_fk_rec["V_roadmap"]),
                     w=np.array(_fk_rec["w"]), step=np.array(_fk_rec["step"]),
                     idx=np.array(_fk_rec["idx"]) if _fk_rec["idx"] else np.zeros((0,)),
                     resampled=np.array(_fk_rec["resampled"]),
                     init_std=self.init_std, exit_step=exit_step,
                     success=current_success)
            print(f"[FK-SMC] dumped particle trace -> {_dump_path_fk}")

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        fname = f"seed_success_chunk_{task_id}.txt" if task_id else "seed_success_log.txt"
        with open(os.path.join(os.environ.get("GRASP_RESULTS_DIR", os.getcwd()), fname), "a") as f:
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {exit_step}\n")

        best_idx = int(torch.argmin(L_dyn.detach()))
        final_action = a[best_idx:best_idx + 1].detach()
        return final_action, np.full(1, np.inf)


    def _aft_iaf_move(self, a, w, obs_batch, zg_batch, P, S, sigma_t, reg_w):
        """One AFT transport: fit a shared flow on the weighted free energy, apply it.

        Returns (L_dyn at the transported particles, mean |displacement|). Used
        either at the bottom of the loop (our order) or at the top
        (transport_first=True: Arbel et al. Alg. 1, transport -> reweight -> resample).
        """
        from .aft_flows import AffineIAF, ResidualFlow
        Pn, Hn, An = a.shape
        if self.flow_type == "resflow":
            mk = lambda: ResidualFlow(Hn * An, hidden=self.iaf_hids * 8, layers=self.iaf_layers,
                                      delta=max(self.iaf_bounded_shift, 0.2)).to(a.device)
        else:
            mk = lambda: AffineIAF(Hn * An, self.iaf_hids, self.iaf_layers,
                                   shift_only=self.iaf_shift_only,
                                   shift_scale=self.iaf_shift_scale,
                                   bounded_shift=self.iaf_bounded_shift).to(a.device)
        # Distillation refits to fresh targets every iteration, so warm-starting the
        # network is safe (and necessary: from a cold identity init the fit cannot
        # reach 32 targets in K steps -- measured derr/tgt = 0.017/0.023 on cube).
        if self.iaf_distill and self.flow_persist and self._flow_state is not None:
            flows, fopt = self._flow_state
        else:
            flows = [mk() for _ in range(Pn)] if self.iaf_per_particle else [mk()]
            params = [q for f_ in flows for q in f_.parameters()]
            fopt = torch.optim.Adam(params, lr=self.flow_lr)
            if self.iaf_distill and self.flow_persist:
                self._flow_state = (flows, fopt)
        flow = flows[0]

        def apply_flows(z):
            if not self.iaf_per_particle:
                return flows[0](z)
            outs = [flows[j](z[j:j + 1]) for j in range(Pn)]
            return torch.cat([o[0] for o in outs], 0), torch.cat([o[1] for o in outs], 0)
        w_fit = w.detach() if self.flow_weighted else torch.ones_like(w)
        x0 = a.detach().reshape(Pn, Hn * An)

        target = None
        if self.iaf_distill:
            # the adam kernel's own update, computed here as the distillation target
            _, g = self._loss_and_grad(a, obs_batch, zg_batch, P, S, sigma_t, reg_w)
            b1, b2, eps = 0.9, 0.999, 1e-8
            if self._dm is None:
                self._dm = torch.zeros_like(g); self._dv = torch.zeros_like(g); self._dt = 0
            self._dt += 1
            self._dm = b1 * self._dm + (1 - b1) * g
            self._dv = b2 * self._dv + (1 - b2) * g * g
            mhat = self._dm / (1 - b1 ** self._dt)
            vhat = self._dv / (1 - b2 ** self._dt)
            d = -self.langevin_lr * mhat / (vhat.sqrt() + eps)
            target = (a.detach() + d).reshape(Pn, Hn * An)

        for _ in range(self.flow_steps):
            y_flat, logdet = apply_flows(x0)
            y = y_flat.reshape(Pn, Hn, An)
            if target is not None:
                free_energy = ((y_flat - target) ** 2).mean(dim=1).mul(w_fit).sum()
                L_y = torch.zeros(Pn, device=a.device)
            else:
                L_y = self._smoothed_loss(y, obs_batch, zg_batch, P, S, sigma_t)
                if reg_w > 0:
                    L_y = L_y + reg_w * (y ** 2).mean(dim=(1, 2))
                free_energy = (w_fit * (L_y - self.flow_logdet_weight * logdet)).sum()
            fopt.zero_grad()
            free_energy.backward()
            fopt.step()
        with torch.no_grad():
            y_flat, logdet = apply_flows(x0)
            a.data = y_flat.reshape(Pn, Hn, An)
            if self.langevin_temp > 0:
                a.data += torch.randn_like(a) * math.sqrt(2 * self.langevin_lr * self.langevin_temp)
            if self.action_clamp > 0:
                a.data.clamp_(-self.action_clamp, self.action_clamp)
            disp = (a.data.reshape(Pn, -1) - x0).abs().mean().item()
            if target is not None:                      # measure the loss we actually reached
                L_y = self._smoothed_loss(a.detach(), obs_batch, zg_batch, P, 1, 0.0)
                self._distill_err = (a.data.reshape(Pn, -1) - target).abs().mean().item()
                self._distill_tgt = (target - x0).abs().mean().item()
        self._last_flow = (float(logdet.mean().item()), disp)
        if not (self.iaf_distill and self.flow_persist):
            del flows, flow, fopt
        return L_y.detach(), disp

    def _smoothed_loss(self, y, obs_batch, zg_batch, P, S, sigma_t):
        """L_dyn at y (P,H,A) averaged over S Gaussian perturbations, WITH graph.
        Same mollified objective the adam kernel descends."""
        if S > 1:
            eps = torch.randn(S, *y.shape, device=y.device) * sigma_t
            y_eval = (y.unsqueeze(0) + eps).flatten(0, 1)
            obs_eval = {k: repeat(v, "p ... -> (s p) ...", s=S) for k, v in obs_batch.items()} \
                if isinstance(obs_batch, dict) else repeat(obs_batch, "p ... -> (s p) ...", s=S)
            zg_eval = {k: repeat(v, "p ... -> (s p) ...", s=S) for k, v in zg_batch.items()}
        else:
            y_eval, obs_eval, zg_eval = y, obs_batch, zg_batch
        i_z, _ = self.wm.rollout(obs_0=obs_eval, act=y_eval)
        L_all = self.objective_fn(i_z, zg_eval, y_eval)
        return L_all.view(S, P).mean(dim=0) if S > 1 else L_all
