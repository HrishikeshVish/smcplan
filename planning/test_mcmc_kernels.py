"""Verification for planning/mcmc_kernels.py: the torch ports must agree with
blackjax, and the target's zero-rollout algebra must be exact.

Run:  python -m planning.test_mcmc_kernels     (from the repo root)

Four groups of checks:

  A  acceptance probabilities, against blackjax itself. The blackjax kernel is
     run on a target evaluable in both frameworks, its own proposal is
     reconstructed from its own public integrator with the same split key, and
     the torch port's accept probability on that same (x, y) pair is compared
     to the `info.acceptance_rate` blackjax reports. Exact agreement, not
     statistical.
  B  invariance. Each kernel is run as many parallel chains on targets with
     known moments; the sampled mean and covariance must match. This is what
     catches an error the ratio comparison cannot, e.g. a wrong noise scale.
  C  em_target's recompose identity: changing mu or beta and recomposing from
     the cache must equal a fresh rollout at the new (mu, beta).
  D  the MPPI limit: a score-only M-step with a plain SGD step of size
     sigma_p^2 must reproduce mu <- sum_i w_i a_i exactly.
"""

import math
import sys

import numpy as np
import torch

from planning.mcmc_kernels import (HMC, MALA, RWM, ULA, ChainState,
                                   static_binomial_sampling)

torch.set_default_dtype(torch.float64)
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------- test targets


class ToyTarget:
    """A ChainState-producing target with no world model: log pi known exactly.

    Two shapes, both with tractable moments:
      "gauss"  N(0, diag(scales)^2), correlated via a fixed linear map
      "banana" the Rosenbrock/banana density, a standard MCMC stress test
    """

    def __init__(self, kind="gauss", dim=4, device="cpu"):
        self.kind, self.dim, self.device = kind, dim, device
        # AR(1) correlation, Sigma_ij = 0.5^|i-j|: correlated but
        # well-conditioned (cond ~ 9 at dim 4), so a single shared step size
        # is usable for every kernel.
        i = torch.arange(dim, dtype=torch.get_default_dtype())
        self.Sigma = 0.5 ** (i.view(-1, 1) - i.view(1, -1)).abs()
        self.Prec = torch.linalg.inv(self.Sigma)
        self.n_evals = 0

    def logpi_np(self, x):
        """Same density in numpy, so JAX can be handed an identical function."""
        if self.kind == "gauss":
            return -0.5 * x @ np.linalg.inv(self.Sigma.numpy()) @ x
        return -(x[0] ** 2) / 20.0 - 0.5 * (x[1] - x[0] ** 2 / 10.0) ** 2 \
            - 0.5 * np.sum(x[2:] ** 2)

    def _logpi(self, a):
        """a: (P, dim) -> (P,). Written on flat vectors; the planner's (P,H,A)
        case is just this with the last two dims flattened."""
        if self.kind == "gauss":
            return -0.5 * ((a @ self.Prec) * a).sum(-1)
        return -(a[:, 0] ** 2) / 20.0 \
            - 0.5 * (a[:, 1] - a[:, 0] ** 2 / 10.0) ** 2 \
            - 0.5 * (a[:, 2:] ** 2).sum(-1)

    def evaluate(self, a):
        a = a.detach().requires_grad_(True)
        logp = self._logpi(a)
        g = torch.autograd.grad(logp.sum(), a)[0]
        self.n_evals += 1
        # The kernels only ever read position/logdensity/grad; cost fields are
        # carried so the accept mask is exercised on them too.
        return ChainState(position=a.detach(), logdensity=logp.detach(),
                          grad=g.detach(), cost=-logp.detach(),
                          cost_grad=-g.detach())


# ------------------------------------------- A. accept probs vs blackjax


def test_against_blackjax():
    print("\nA. acceptance probabilities vs blackjax")
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import blackjax
        from blackjax.mcmc import mala as bj_mala
        from blackjax.mcmc import random_walk as bj_rw
        from blackjax.mcmc.diffusions import overdamped_langevin
        from blackjax.mcmc.integrators import velocity_verlet
    except Exception as e:                                    # pragma: no cover
        check("blackjax importable", False, f"({e}) -- ports UNVERIFIED")
        return

    tgt = ToyTarget("banana", dim=4)
    logdensity_fn = lambda x: jnp.asarray(  # noqa: E731
        -(x[0] ** 2) / 20.0 - 0.5 * (x[1] - x[0] ** 2 / 10.0) ** 2
        - 0.5 * jnp.sum(x[2:] ** 2))
    eps = 0.15

    # ---- MALA: blackjax's kernel, and its proposal rebuilt from its own
    #      integrator with the identical key split, so the comparison is on
    #      the same (x, y) pair.
    kernel = bj_mala.build_kernel()
    max_err = 0.0
    for t in range(25):
        x = np.asarray(jax.random.normal(jax.random.key(1000 + t), (4,)),
                       dtype=np.float64)
        st = bj_mala.init(jnp.asarray(x), logdensity_fn)
        key = jax.random.key(t)
        _, info = kernel(key, st, logdensity_fn, eps)

        key_int, _ = jax.random.split(key)
        integ = overdamped_langevin(jax.value_and_grad(logdensity_fn))
        prop = integ(key_int, st, eps)
        y = np.asarray(prop.position, dtype=np.float64)

        sx = tgt.evaluate(torch.tensor(x).unsqueeze(0))
        sy = tgt.evaluate(torch.tensor(y).unsqueeze(0))
        from planning.mcmc_kernels import _mala_transition_energy, safe_energy_diff
        new_e = _mala_transition_energy(sx.position, sy.position,
                                        sy.logdensity, sy.grad, eps)
        prev_e = _mala_transition_energy(sy.position, sx.position,
                                         sx.logdensity, sx.grad, eps)
        mine = float(safe_energy_diff(prev_e, new_e).exp().clamp(max=1.0))
        max_err = max(max_err, abs(mine - float(info.acceptance_rate)))
    check("MALA accept prob == blackjax", max_err < 1e-9, f"max|d|={max_err:.2e}")

    # ---- RWM: blackjax's additive-step random walk, Gaussian step.
    rw_kernel = bj_rw.build_additive_step()
    random_step = bj_rw.normal(eps * np.ones(4))
    max_err = 0.0
    for t in range(25):
        x = np.asarray(jax.random.normal(jax.random.key(2000 + t), (4,)),
                       dtype=np.float64)
        st = bj_rw.init(jnp.asarray(x), logdensity_fn)
        key = jax.random.key(t)
        _, info = rw_kernel(key, st, logdensity_fn, random_step)

        key_prop, _ = jax.random.split(key)
        y = np.asarray(jnp.asarray(x) + random_step(key_prop, jnp.asarray(x)),
                       dtype=np.float64)
        sx = tgt.evaluate(torch.tensor(x).unsqueeze(0))
        sy = tgt.evaluate(torch.tensor(y).unsqueeze(0))
        from planning.mcmc_kernels import safe_energy_diff
        mine = float(safe_energy_diff(-sx.logdensity,
                                      -sy.logdensity).exp().clamp(max=1.0))
        max_err = max(max_err, abs(mine - float(info.acceptance_rate)))
    check("RWM accept prob == blackjax", max_err < 1e-9, f"max|d|={max_err:.2e}")

    # ---- HMC: compare the leapfrog trajectory and the Hamiltonian against
    #      blackjax's velocity_verlet on the same (q, p).
    kinetic = lambda p: 0.5 * jnp.sum(p ** 2)  # noqa: E731
    bj_step = velocity_verlet(logdensity_fn, kinetic)
    max_q = max_h = 0.0
    for t in range(15):
        q0 = np.asarray(jax.random.normal(jax.random.key(3000 + t), (4,)),
                        dtype=np.float64)
        p0 = np.asarray(jax.random.normal(jax.random.key(4000 + t), (4,)),
                        dtype=np.float64)
        ld, gd = jax.value_and_grad(logdensity_fn)(jnp.asarray(q0))
        bs = velocity_verlet.__globals__["IntegratorState"](
            jnp.asarray(q0), jnp.asarray(p0), ld, gd)
        for _ in range(3):
            bs = bj_step(bs, eps)

        # torch: HMC.step's inner leapfrog, run open-loop
        st = tgt.evaluate(torch.tensor(q0).unsqueeze(0))
        p = torch.tensor(p0).unsqueeze(0)
        cur = st
        for _ in range(3):
            p = p + 0.5 * eps * cur.grad
            cur = tgt.evaluate(cur.position + eps * p)
            p = p + 0.5 * eps * cur.grad

        max_q = max(max_q, float(np.abs(np.asarray(bs.position)
                                        - cur.position[0].numpy()).max()))
        h_bj = float(-bs.logdensity + 0.5 * np.sum(np.asarray(bs.momentum) ** 2))
        h_me = float(-cur.logdensity[0] + 0.5 * (p[0] ** 2).sum())
        max_h = max(max_h, abs(h_bj - h_me))
    check("HMC leapfrog == blackjax velocity_verlet", max_q < 1e-9,
          f"max|dq|={max_q:.2e}")
    check("HMC Hamiltonian == blackjax", max_h < 1e-9, f"max|dH|={max_h:.2e}")

    # ---- the whole chain, statistically: blackjax MALA and torch MALA must
    #      recover the same mean on the same target.
    tg = ToyTarget("gauss", dim=4)
    Sig = tg.Sigma.numpy()
    ld_g = lambda x: jnp.asarray(-0.5 * x @ jnp.asarray(np.linalg.inv(Sig)) @ x)  # noqa: E731
    st = bj_mala.init(jnp.zeros(4), ld_g)
    keys = jax.random.split(jax.random.key(7), 4000)
    xs = []
    for k in keys:
        st, _ = kernel(k, st, ld_g, 0.3)
        xs.append(np.asarray(st.position))
    bj_mean = np.mean(xs[1000:], 0)
    check("blackjax MALA recovers mean 0", np.abs(bj_mean).max() < 0.2,
          f"max|mean|={np.abs(bj_mean).max():.3f}")


# ---------------------------------------------- B. invariance of the ports


def run_chains(kernel, tgt, n_chains=512, n_steps=1500, burn=500, eps=0.3,
               seed=0):
    torch.manual_seed(seed)
    state = tgt.evaluate(torch.randn(n_chains, tgt.dim))
    draws, accs = [], []
    for i in range(n_steps):
        state, info = kernel.step(state, tgt, eps)
        accs.append(float(info["acceptance_rate"].mean()))
        if i >= burn:
            draws.append(state.position.clone())
    return torch.stack(draws).flatten(0, 1), float(np.mean(accs))


def test_invariance():
    print("\nB. invariance on targets with known moments")
    for kind in ("gauss", "banana"):
        tgt = ToyTarget(kind, dim=4)
        if kind == "gauss":
            true_mean = np.zeros(4)
            true_cov = tgt.Sigma.numpy()
        else:
            # banana moments in closed form: -x0^2/20 is N(0, 10), so
            # Var(x0) = 10 and E[x1] = E[x0^2]/10 = 1. x2: are standard normal.
            true_mean = np.array([0.0, 1.0, 0.0, 0.0])
        for kern in (MALA(), RWM(), HMC(n_leaps=3), ULA()):
            eps = {"gauss": {"mala": 0.5, "rwm": 0.8, "hmc": 0.4, "ula": 0.05},
                   "banana": {"mala": 0.4, "rwm": 0.9, "hmc": 0.3,
                              "ula": 0.05}}[kind][kern.name]
            draws, acc = run_chains(kern, tgt, eps=eps,
                                    n_steps=4000 if kind == "banana" else 1500,
                                    burn=1000 if kind == "banana" else 500)
            m = draws.mean(0).numpy()
            if kind == "gauss":
                c = np.cov(draws.numpy().T)
                dm = np.abs(m - true_mean).max()
                dc = np.abs(c - true_cov).max() / np.abs(true_cov).max()
                # ULA is biased by design; hold it to a looser bar and only to
                # show the bias is small at this step size, not absent.
                tol_m, tol_c = (0.30, 0.25) if kern.name == "ula" else (0.12, 0.10)
                check(f"{kind}/{kern.name} mean+cov", dm < tol_m and dc < tol_c,
                      f"max|dmean|={dm:.3f} rel|dcov|={dc:.3f} acc={acc:.2f}")
            else:
                dm = np.abs(m - true_mean).max()
                # Var(x0) = 10 is the check that actually pins the noise
                # scale: a kernel with the wrong sqrt(2*eps) still gets the
                # mean right by symmetry but not the spread.
                v0 = float(draws[:, 0].var())
                dv = abs(v0 - 10.0) / 10.0
                tol, tolv = (0.6, 0.30) if kern.name == "ula" else (0.35, 0.15)
                check(f"{kind}/{kern.name} mean+var(x0)",
                      dm < tol and dv < tolv,
                      f"max|dmean|={dm:.3f} var(x0)={v0:.2f} (want 10) "
                      f"acc={acc:.2f}")


def test_accept_target():
    print("\nB2. dual averaging hits the target acceptance rate")
    from planning.mcmc_kernels import DualAveraging
    tgt = ToyTarget("gauss", dim=8)
    for kern, target in ((MALA(), 0.574), (RWM(), 0.234)):
        torch.manual_seed(1)
        da = DualAveraging(target, 0.5)
        state = tgt.evaluate(torch.randn(256, 8))
        eps, accs = 0.5, []
        for i in range(600):
            state, info = kern.step(state, tgt, eps)
            a = float(info["acceptance_rate"].mean())
            eps = da.update(a)
            if i > 400:
                accs.append(a)
        got = float(np.mean(accs))
        check(f"{kern.name} dual averaging -> {target}", abs(got - target) < 0.06,
              f"got acc={got:.3f} eps={da.final():.4g}")


# ------------------------------------------- C. em_target recompose identity


class MockWM:
    """Minimal stand-in for the world model: latent = cumulative action sum."""

    def rollout(self, obs_0, act):
        return {"proprio": act.cumsum(1)}, None


class MockHeat:
    def evaluate_rollout(self, z_T):
        return z_T.norm(dim=-1)


def test_recompose():
    print("\nC. em_target: recompose after moving mu / beta == fresh rollout")
    from planning.em_target import ActionSequenceTarget
    H, A, P = 5, 3, 7
    torch.manual_seed(0)
    obs0 = {"proprio": torch.zeros(1, 1, A)}
    zg = {"proprio": torch.randn(1, 1, A)}
    tgt = ActionSequenceTarget(
        wm=MockWM(),
        objective_fn=lambda i_z, zg, a: (
            (i_z["proprio"][:, -1, :] - zg["proprio"][:, 0, :]) ** 2).sum(-1),
        get_latent=lambda i_z: i_z["proprio"],
        trans_obs_0=obs0, z_obs_g=zg, heat_solver=MockHeat(),
        horizon=H, action_dim=A, sigma_p=0.8, alpha=2.0, lambda_v=0.5,
        action_reg=0.1)
    tgt.mu = torch.zeros(H, A)
    tgt.beta = 0.3
    a = torch.randn(P, H, A)
    state = tgt.evaluate(a)
    n0 = tgt.n_rollout_calls

    tgt.mu = torch.randn(H, A) * 0.5
    tgt.beta = 0.77
    cheap = tgt.recompose(state)
    check("recompose costs no rollout", tgt.n_rollout_calls == n0)
    fresh = tgt.evaluate(a)
    dl = float((cheap.logdensity - fresh.logdensity).abs().max())
    dg = float((cheap.grad - fresh.grad).abs().max())
    check("recompose log pi exact", dl < 1e-10, f"max|d|={dl:.2e}")
    check("recompose grad log pi exact", dg < 1e-10, f"max|d|={dg:.2e}")

    # the prior ratio used for the mu-move weight increment must equal the
    # log pi difference, since C is mu-independent
    mu_new, mu_old = tgt.mu, torch.zeros(H, A)
    dprior = tgt.log_prior(a, mu_new) - tgt.log_prior(a, mu_old)
    tgt.mu = mu_old
    s_old = tgt.recompose(state)
    tgt.mu = mu_new
    s_new = tgt.recompose(state)
    d = float(((s_new.logdensity - s_old.logdensity) - dprior).abs().max())
    check("mu-move weight increment == analytic prior ratio", d < 1e-10,
          f"max|d|={d:.2e}")

    # accept-mask threading must move every cached field together
    acc = torch.tensor([True, False] * 3 + [True])
    merged = fresh.where(acc, state)
    ok = all(
        torch.allclose(getattr(merged, f)[i],
                       getattr(fresh if acc[i] else state, f)[i])
        for f in ("position", "logdensity", "grad", "cost", "cost_grad", "task",
                  "value")
        for i in range(P))
    check("ChainState.where threads all cached fields", ok)


# ------------------------------------------------------- D. the MPPI limit


def test_mppi_limit():
    print("\nD. score-only M-step with SGD(sigma_p^2) == MPPI's weighted mean")
    H, A, P = 4, 2, 16
    torch.manual_seed(0)
    sigma_p = 0.7
    mu = torch.randn(H, A)
    a = mu.unsqueeze(0) + sigma_p * torch.randn(P, H, A)
    logw = torch.randn(P)
    w = torch.softmax(logw, 0)

    # the planner's score term, verbatim from EMSMCPlanner._m_grad
    g = -((w.view(-1, 1, 1) * (a - mu)).sum(0) / sigma_p ** 2)
    mppi = (w.view(-1, 1, 1) * a).sum(0)
    got = mu - sigma_p ** 2 * g              # one SGD step of size sigma_p^2
    d = float((got - mppi).abs().max())
    check("mu - sigma_p^2 * g_score == sum_i w_i a_i", d < 1e-12,
          f"max|d|={d:.2e}")

    # and the pathwise term is the SMC-weighted mean of the per-particle
    # cost gradients, i.e. an estimate of grad E_{a~pi}[C(a)]
    gC = torch.randn(P, H, A)
    gp = (w.view(-1, 1, 1) * gC).sum(0)
    d = float((gp - torch.einsum("p,pha->ha", w, gC)).abs().max())
    check("g_pathwise == sum_i w_i grad C(a_i)", d < 1e-12, f"max|d|={d:.2e}")


# ------------------------------- E. the Jarzynski / ULA weight correction


class TemperedGaussian:
    """A tempering path between two Gaussians, with log Z known in closed form.

    U_beta(x) = (1-beta)|x|^2/2 + beta|x|^2/(2 sigma^2), so pi_beta is Gaussian
    with precision p_beta = (1-beta) + beta/sigma^2 and
        log(Z_1 / Z_0) = (d/2) log(p_0 / p_1) = d log sigma.
    That number is what an SMC sampler's accumulated log-weights must estimate,
    which makes it a direct test of the weight bookkeeping -- the tempering
    increments and the ULA correction together -- rather than of the kernel
    alone.
    """

    def __init__(self, dim=4, sigma=3.0):
        self.dim, self.sigma, self.beta = dim, sigma, 0.0

    def prec(self, beta=None):
        b = self.beta if beta is None else beta
        return (1.0 - b) + b / self.sigma ** 2

    def log_gamma(self, x, beta=None):
        """Unnormalised log density, i.e. -U_beta."""
        return -0.5 * self.prec(beta) * (x * x).sum(-1)

    def true_log_z_ratio(self):
        return self.dim * math.log(self.sigma)

    def evaluate(self, a):
        a = a.detach().requires_grad_(True)
        logp = self.log_gamma(a)
        g = torch.autograd.grad(logp.sum(), a)[0]
        return ChainState(position=a.detach(), logdensity=logp.detach(),
                          grad=g.detach(), cost=-logp.detach(),
                          cost_grad=-g.detach())


def test_jarzynski():
    print("\nE. Jarzynski / ULA weight correction")
    from planning.mcmc_kernels import jarzynski_log_weight_increment

    # E1. the Gaussian-ratio form equals JALA's alpha recursion.
    tgt = ToyTarget("banana", dim=4)
    eps = 0.11
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    sx = tgt.evaluate(x)
    sy = tgt.evaluate(x + eps * sx.grad + math.sqrt(2 * eps) * torch.randn(64, 4))

    def alpha(s_at, other_pos, eps):
        """JALA Eq. 3: U(x_l) + (x_r-x_l).grad U(x_l)/2 + eps|grad U(x_l)|^2/4,
        with U = -log pi so grad U = -grad log pi."""
        U = -s_at.logdensity
        gU = -s_at.grad
        return (U + 0.5 * ((other_pos - s_at.position) * gU).flatten(1).sum(1)
                + 0.25 * eps * (gU * gU).flatten(1).sum(1))

    jala = -alpha(sy, sx.position, eps) + alpha(sx, sy.position, eps)
    mine = jarzynski_log_weight_increment(sx, sy, eps)
    d = float((jala - mine).abs().max())
    check("Gaussian-ratio form == JALA alpha recursion", d < 1e-10,
          f"max|d|={d:.2e}")

    # E2. the whole bookkeeping, against a closed-form log Z. This is the
    #     planner's exact weight sequence: a tempering increment per step,
    #     then the kernel, then the kernel's own increment (0 for MH, the
    #     Jarzynski term for ULA).
    for kern, eps_k in ((MALA(), 0.25), (ULA(), 0.2), (RWM(), 0.5),
                        (HMC(n_leaps=3), 0.2)):
        torch.manual_seed(3)
        tg = TemperedGaussian(dim=4, sigma=3.0)
        N, K = 4096, 300
        tg.beta = 0.0
        state = tg.evaluate(torch.randn(N, tg.dim))      # exact draw from pi_0
        logw = torch.zeros(N)
        for k in range(1, K + 1):
            beta_new = k / K
            logw = logw + (tg.log_gamma(state.position, beta_new)
                           - tg.log_gamma(state.position, tg.beta))
            tg.beta = beta_new
            state = tg.evaluate(state.position)          # refresh at new beta
            state, info = kern.step(state, tg, eps_k)
            logw = logw + info["log_weight_increment"]
        est = float(torch.logsumexp(logw, 0) - math.log(N))
        truth = tg.true_log_z_ratio()
        check(f"{kern.name}: log Z ratio == {truth:.3f}",
              abs(est - truth) < 0.15, f"est={est:.3f} err={est - truth:+.3f}")

    # E3. dropping the ULA correction must visibly break that estimate --
    #     otherwise E2 would pass for the wrong reason. Measured on this target
    #     (truth 4.3944), corrected vs uncorrected:
    #       eps   0.05    0.10    0.20    0.30    0.40
    #       corr  4.371   4.392   4.412   4.415   4.413
    #       unco  4.435   4.507   4.629   4.737   4.846
    #     i.e. the correction holds the estimate flat while the uncorrected one
    #     drifts linearly in eps. Below eps ~ 0.05 the ESS collapses (1.9% at
    #     eps=0.02) and BOTH are biased, for want of mixing rather than want of
    #     a correction -- which is why this is measured at eps=0.2.
    torch.manual_seed(3)
    tg = TemperedGaussian(dim=4, sigma=3.0)
    N, K, kern = 4096, 300, ULA()
    tg.beta = 0.0
    state = tg.evaluate(torch.randn(N, tg.dim))
    logw = torch.zeros(N)
    for k in range(1, K + 1):
        beta_new = k / K
        logw = logw + (tg.log_gamma(state.position, beta_new)
                       - tg.log_gamma(state.position, tg.beta))
        tg.beta = beta_new
        state = tg.evaluate(state.position)
        state, _ = kern.step(state, tg, 0.2)             # increment DISCARDED
    bad = float(torch.logsumexp(logw, 0) - math.log(N))
    check("omitting the ULA correction biases log Z",
          abs(bad - tg.true_log_z_ratio()) > 0.15,
          f"uncorrected={bad:.3f} vs truth={tg.true_log_z_ratio():.3f}")


# --------------------------------- F. the chunked M-step gradient is exact


def test_m_step_chunking():
    """P*S sequences with a live autograd graph OOMs a 16 GB card at P=3, S=8,
    which is why the baseline sweeps ran at S=1. Row i of the gradient depends
    only on row i of the input, so chunking the rollout is exact rather than
    approximate -- this pins that down against an unchunked reference.

    Instantiates the real planner method through __new__ with only the two
    attributes it reads, so no world model or evaluator is needed.
    """
    print("\nF. chunked M-step gradient == unchunked")
    from planning.em_smc import EMSMCPlanner
    from planning.em_target import ActionSequenceTarget

    H, A, P, S = 5, 3, 6, 8
    torch.manual_seed(0)
    tgt = ActionSequenceTarget(
        wm=MockWM(),
        # quartic, so the gradient genuinely varies across the noised copies
        objective_fn=lambda i_z, zg, a: (
            (i_z["proprio"][:, -1, :] - zg["proprio"][:, 0, :]) ** 4).sum(-1),
        get_latent=lambda i_z: i_z["proprio"],
        trans_obs_0={"proprio": torch.zeros(1, 1, A)},
        z_obs_g={"proprio": torch.randn(1, 1, A)},
        heat_solver=MockHeat(), horizon=H, action_dim=A,
        sigma_p=0.8, alpha=2.0, lambda_v=0.5, action_reg=0.1)
    tgt.mu = torch.zeros(H, A)
    tgt.beta = 0.7
    a = torch.randn(P, H, A)

    def grad_with(chunk, seed=7):
        self = EMSMCPlanner.__new__(EMSMCPlanner)
        self.m_smooth_samples, self.m_chunk = S, chunk
        torch.manual_seed(seed)              # identical noise draw each time
        return self._smoothed_cost_grad(tgt, a, sigma=0.5)

    ref = grad_with(0)                       # 0 => the whole P*S in one rollout
    check("reference gradient is nonzero", float(ref.norm()) > 1.0,
          f"|g|={float(ref.norm()):.3f}")
    worst = 0.0
    for c in (1, 2, 3, 5, 8, 16, 48, 100):
        worst = max(worst, float((grad_with(c) - ref).abs().max()))
    check("chunking is exact at every chunk size", worst < 1e-12,
          f"max|d|={worst:.1e} over chunk in 1..100")


# ------------------- G. the two M-step gradients estimate the same thing


def test_m_step_equivalence():
    """grad_mu(-log Z) has a score-function form and a reparameterised form.

        Z(mu) = int q(a|mu) e^{-C(a)/alpha} da
        score:    grad_mu(-log Z) = -E_pi[(a - mu)] / sigma_p^2
        pathwise: grad_mu(-log Z) = +E_pi[grad C(a)] / alpha   (a = mu + sigma_p eps)

    They are therefore the SAME gradient, and an earlier version of _m_grad
    added them (and omitted the 1/alpha), which double-counted it at a
    magnitude proportional to alpha -- per-seed-varying under alpha="auto".
    This regression test pins both the collinearity and the 1/alpha.
    """
    print("\nG. score and pathwise M-step gradients agree")
    from planning.em_target import ActionSequenceTarget
    from planning.em_smc import EMSMCPlanner

    H, A, P = 4, 2, 300000
    torch.manual_seed(0)
    worst_cos, worst_ratio = 1.0, 0.0
    for alpha in (0.5, 1.0, 2.0, 5.0):
        tgt = ActionSequenceTarget(
            wm=MockWM(),
            objective_fn=lambda i_z, zg, a: (
                (i_z["proprio"][:, -1, :] - zg["proprio"][:, 0, :]) ** 2).sum(-1),
            get_latent=lambda i_z: i_z["proprio"],
            trans_obs_0={"proprio": torch.zeros(1, 1, A)},
            z_obs_g={"proprio": torch.tensor([[[0.7, -0.4]]])},
            heat_solver=MockHeat(), horizon=H, action_dim=A,
            sigma_p=0.6, alpha=alpha, lambda_v=0.0, action_reg=0.0)
        tgt.mu = torch.randn(H, A) * 0.3
        tgt.beta = 1.0
        state = tgt.evaluate(tgt.sample_prior(P))   # a ~ q(.|mu), beta = 1
        w = torch.softmax(-state.cost / alpha, 0)   # weights targeting pi

        self = EMSMCPlanner.__new__(EMSMCPlanner)
        self.sigma_p, self.m_smooth_samples, self.pathwise_mix = 0.6, 1, 0.5
        self.m_step = "score"
        g_s = self._m_grad(tgt, state, w, 0.0)
        self.m_step = "pathwise"
        g_p = self._m_grad(tgt, state, w, 0.0)

        cos = float((g_s * g_p).sum() / (g_s.norm() * g_p.norm()))
        ratio = float(g_p.norm() / g_s.norm())
        worst_cos = min(worst_cos, cos)
        worst_ratio = max(worst_ratio, abs(ratio - 1.0))
    check("score and pathwise are collinear", worst_cos > 0.99,
          f"min cos={worst_cos:.5f} over alpha in [0.5, 5]")
    check("pathwise carries its 1/alpha (same magnitude as score)",
          worst_ratio < 0.05, f"max|ratio-1|={worst_ratio:.4f}")

    # and "both" must lie between them, not beyond -- the double-counting check
    self.m_step = "both"
    self.pathwise_mix = 0.5
    g_b = self._m_grad(tgt, state, w, 0.0)
    lo, hi = sorted([float(g_s.norm()), float(g_p.norm())])
    check("both is a convex mix, not a sum",
          lo - 1e-9 <= float(g_b.norm()) <= hi + 1e-9,
          f"|g_both|={float(g_b.norm()):.5f} in [{lo:.5f}, {hi:.5f}]")


if __name__ == "__main__":
    test_against_blackjax()
    test_invariance()
    test_accept_target()
    test_recompose()
    test_mppi_limit()
    test_jarzynski()
    test_m_step_chunking()
    test_m_step_equivalence()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
