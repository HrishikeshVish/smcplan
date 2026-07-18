"""
Verify heat equation on lewm_reacher_buffer.

Loads the buffer, builds the graph, solves for random goals,
and prints diagnostics: V distribution, Spearman correlation,
PCA explained variance, and surrogate MLP quality.
"""

import torch
import numpy as np
from planning.gd import MCMollifiedHeat, _load_replay_buffer
from scipy.stats import spearmanr

device = 'cpu'
buf_path = "/home/hviswan/brainstorming/grasp_langevin/lewm_reacher_buffer.pt"

print("=" * 60)
print("Loading buffer...")
loaded = torch.load(buf_path, map_location='cpu', weights_only=True)
print(f"Keys: {list(loaded.keys())}")
for k, v in loaded.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        print(f"    range=[{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}, std={v.std():.4f}")

has_pca = 'pca_components' in loaded
if has_pca:
    print(f"\nPCA info present:")
    print(f"  components: {loaded['pca_components'].shape}")
    print(f"  mean: {loaded['pca_mean'].shape}")
    pca_dim = loaded['z_curr'].shape[1]
    orig_dim = loaded['pca_mean'].shape[0]
    print(f"  PCA dim: {pca_dim}, original dim: {orig_dim}")

    # Check effective dimensionality of PCA-projected data
    z_all = torch.cat([loaded['z_curr'], loaded['z_next']], dim=0).numpy()
    cov = np.cov(z_all.T)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    cumvar = np.cumsum(eigvals) / eigvals.sum()
    print(f"  Cumulative variance in PCA space: {cumvar}")


def test_heat_solver(z_curr_np, z_next_np, label, n_goals=5):
    """Build graph and test heat solve on multiple goals."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {label}")
    print(f"  z_curr: {z_curr_np.shape}, z_next: {z_next_np.shape}")

    solver = MCMollifiedHeat()
    solver.compile_environment(z_buffer=z_curr_np, z_predicted_next=z_next_np)

    print(f"  sigma_perturb: {solver.sigma_perturb:.6f}")

    rng = np.random.RandomState(42)
    goal_indices = rng.choice(z_curr_np.shape[0], n_goals, replace=False)
    spearmans = []

    for gi, goal_idx in enumerate(goal_indices):
        goal = solver.z_buffer_np[goal_idx]
        goal_raw = solver.scaler_obj.inverse_transform(goal.reshape(1, -1)).flatten()

        # Reset surrogate so it can retrain
        solver.v_surrogate = None
        for p in solver.graph_nodes:
            pass  # graph_nodes don't need grad reset

        V = solver.solve_for_goal(goal_raw)
        V_np = V.cpu().numpy()

        euclid_dists = np.linalg.norm(solver.z_buffer_np - solver.z_buffer_np[solver.goal_idx], axis=1)
        corr, pval = spearmanr(V_np, euclid_dists)
        spearmans.append(corr)

        V_at_goal = V_np[solver.goal_idx]
        print(f"\n  Goal {gi} (idx={goal_idx}):")
        print(f"    V at goal: {V_at_goal:.4f}")
        print(f"    V range: [{V_np.min():.2f}, {V_np.max():.2f}], mean={V_np.mean():.2f}, std={V_np.std():.2f}")
        print(f"    Spearman(V, euclid_dist): {corr:.4f} (p={pval:.2e})")

        # Surrogate quality
        with torch.no_grad():
            V_pred = solver.v_surrogate(solver.graph_nodes).squeeze(-1)
            mse = ((V_pred - solver.V_graph) ** 2).mean().item()
            corr_surr, _ = spearmanr(V_pred.cpu().numpy(), V_np)
        print(f"    Surrogate: MSE={mse:.6f}, Spearman={corr_surr:.4f}")

        # Gradient quality
        sample_idx = rng.choice(z_curr_np.shape[0], 100, replace=False)
        # Need to re-enable grad for surrogate
        for p in solver.v_surrogate.parameters():
            p.requires_grad = True
        z_sample = solver.graph_nodes[sample_idx].clone().requires_grad_(True)
        with torch.enable_grad():
            V_sample = solver.v_surrogate(z_sample).squeeze(-1)
            grads = torch.autograd.grad(V_sample.sum(), z_sample)[0]
        grad_norms = grads.norm(dim=-1)
        dead_grads = (grad_norms < 1e-6).float().mean().item()
        print(f"    Grad norms: mean={grad_norms.mean().item():.4f}, dead={dead_grads*100:.1f}%")
        # Re-freeze
        for p in solver.v_surrogate.parameters():
            p.requires_grad = False

    print(f"\n  Mean Spearman: {np.mean(spearmans):.4f} (+/- {np.std(spearmans):.4f})")
    return spearmans


# Test with PCA buffer (as-is)
print("\n" + "=" * 60)
print("Loading PCA-projected buffer...")
z_curr_np, z_next_np, pca_info = _load_replay_buffer(buf_path, device)
pca_spearmans = test_heat_solver(z_curr_np, z_next_np, f"PCA-projected (dim={z_curr_np.shape[1]})")

# Test without PCA: reconstruct original latents
if has_pca:
    print("\n" + "=" * 60)
    print("Reconstructing original latents (undo PCA)...")
    components = loaded['pca_components'].numpy()  # (pca_dim, D)
    mean = loaded['pca_mean'].numpy()               # (D,)
    z_curr_orig = loaded['z_curr'].numpy() @ components + mean  # (N, D)
    z_next_orig = loaded['z_next'].numpy() @ components + mean  # (N, D)
    print(f"  Reconstructed: z_curr={z_curr_orig.shape}, z_next={z_next_orig.shape}")
    orig_spearmans = test_heat_solver(z_curr_orig, z_next_orig, f"Reconstructed original (dim={z_curr_orig.shape[1]})")

    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"  PCA (dim={z_curr_np.shape[1]}): mean Spearman = {np.mean(pca_spearmans):.4f}")
    print(f"  Reconstructed (dim={z_curr_orig.shape[1]}): mean Spearman = {np.mean(orig_spearmans):.4f}")

print("\nDone.")
