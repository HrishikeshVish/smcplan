"""AIS/SMC-sampler (Feynman-Kac family) planner: a GRASP-style Langevin
kernel over action-sequence particles, tilted toward low heat-solver
cost-to-go V by adaptive-tempered importance weights + systematic resampling.

Design and derivation: see ~/.claude/plans/misty-soaring-gosling.md and the
`fk-smc-planner-design` / `heat-value-equals-linear-bias` memories. The
2026-08-21 rewrite fixes three no-op bugs found in the first version (all
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

        # Telescoped SMC-sampler weights: logw = anchor - beta*V(now), with
        # anchor_i = beta_at_last_resample * V_i(at last resample). Covers
        # both the tempering increment and the reversible-kernel mutation
        # correction (see module docstring). beta starts at 0 <=> logw = -b*V.
        anchor = torch.zeros(P, device=self.device)
        beta = 0.0
        ess_target = self.ess_target_frac * P

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

            # --- V at current particles: forward-eval only, no grad ---
            with torch.no_grad():
                i_z_obses, _ = self.wm.rollout(obs_0=obs_batch, act=a)
                z_T = self._get_latent(i_z_obses)[:, -1, :]
                V_val = self.heat_solver.evaluate_rollout(z_T).detach()

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
            # Instantaneous tilt effect: weighted-minus-uniform mean of V.
            delta_EV = float((w * V_val).sum().item() - V_val.mean().item())

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
            if (not anneal_done) and (ess < ess_target or budget_limited_advance or completing_jump):
                kl = selection_kl(logw)
                with torch.no_grad():
                    idx = systematic_resample(logw)
                    a.data = a.data[idx].clone()
                    anchor = beta * V_val[idx]
                cum_kl += kl
                n_resamples += 1
                last_unique = int(idx.unique().numel())
                # Adam's per-particle state is tied to tensor position;
                # resampling reshuffles particle identity, so reset it.
                optimizer = torch.optim.Adam([a], lr=self.langevin_lr)

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
            for _ in range(self.num_langevin_steps):
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
                    noise = torch.randn_like(a) * math.sqrt(2 * self.langevin_lr * self.langevin_temp)
                    a.data += noise
                    if self.action_clamp > 0:
                        a.data.clamp_(-self.action_clamp, self.action_clamp)

            total_measured_time += time.perf_counter() - start

            # Selection is argmin L_dyn over the FK-tilted population: the
            # tilt acts only through sampling (resampling reshaped which
            # particles exist), never as an objective term. V_sel-vs-pop
            # shows whether the loss winner is also a low-V particle.
            best_idx = int(torch.argmin(L_dyn.detach()))

            if i % self.eval_every == 0:
                print(f"[FK-SMC] step={i:4d} | beta={beta:.3f} | L_dyn={L_dyn[best_idx].item():.4f} "
                      f"| V_sel={V_diag[best_idx].item():.4f} (pop {V_val.mean().item():.2f}) "
                      f"| ESS={ess:.1f}/{P} "
                      f"| dEV={delta_EV:+.3f} | resamples={n_resamples} (uniq={last_unique}) "
                      f"| cumKL={cum_kl:.2f} "
                      f"| cos_sim={cos_sim_val:.4f} | dmin={dmin_val:.4f}")

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

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        fname = f"seed_success_chunk_{task_id}.txt" if task_id else "seed_success_log.txt"
        with open(os.path.join(os.environ.get("GRASP_RESULTS_DIR", os.getcwd()), fname), "a") as f:
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {exit_step}\n")

        best_idx = int(torch.argmin(L_dyn.detach()))
        final_action = a[best_idx:best_idx + 1].detach()
        return final_action, np.full(1, np.inf)
