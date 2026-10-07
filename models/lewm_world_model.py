"""
Wrapper that adapts a LeWM JEPA model to the VWorldModel-compatible interface
expected by grasp_langevin planners.

LeWM operates on CLS-token embeddings (B, T, D) with no patch dimension,
while VWorldModel uses (B, T, num_patches, D). This wrapper bridges the gap
by treating the CLS token as a single "patch" of dimension D.
"""

import torch
import torch.nn as nn
from einops import rearrange


class LeWMWorldModel(nn.Module):
    """Adapter from LeWM JEPA to grasp_langevin planner interface.

    The planners expect a world model with:
      - rollout(obs_0, act) -> (z_obses_dict, z_tensor)
      - encode_obs(obs) -> dict with 'visual', 'proprio'
      - encode_observation_to_vector(obs) -> flat tensor
      - next_state(z_state, actions) -> z_next
    """

    def __init__(
        self,
        jepa_model,
        history_size=3,
        frameskip=5,
        action_dim=2,
        proprio_dim=0,
        image_size=224,
    ):
        super().__init__()
        self.jepa = jepa_model
        self.history_size = history_size
        self.frameskip = frameskip
        self.raw_action_dim = action_dim
        self.action_dim = action_dim  # used by planners (single-step)
        self.proprio_dim = proprio_dim
        self.image_size = image_size

        # Infer embedding dimension from the projector or encoder
        self.emb_dim = self._infer_emb_dim()

        # num_hist / num_pred used by VWorldModel interface
        self.num_hist = history_size
        self.num_pred = 1

    def _infer_emb_dim(self):
        """Infer embedding dimension from model parameters."""
        for p in self.jepa.projector.parameters():
            if p.ndim == 2:
                return p.shape[0]
        # Fallback: try encoder
        for p in self.jepa.encoder.parameters():
            if p.ndim == 2:
                return p.shape[-1]
        raise ValueError("Could not infer embedding dimension from JEPA model")

    def _obs_to_pixels(self, obs):
        """Convert VWorldModel obs dict to JEPA pixels format.

        VWorldModel: obs['visual'] is (B, T, C, H, W)
        JEPA:        info['pixels'] is (B, T, C, H, W)
        """
        if isinstance(obs, dict):
            return obs['visual']
        return obs

    @torch.no_grad()
    def encode_obs(self, obs):
        """Encode observations to embedding dict.

        Args:
            obs: dict with 'visual' (B, T, C, H, W) and 'proprio' (B, T, D)

        Returns:
            dict with 'visual' (B, T, 1, emb_dim) and 'proprio' (B, T, D)
        """
        pixels = self._obs_to_pixels(obs)
        info = {'pixels': pixels}
        info = self.jepa.encode(info)
        emb = info['emb']  # (B, T, D)

        result = {
            'visual': emb.unsqueeze(2),  # (B, T, 1, D) - single "patch"
            'proprio': obs.get('proprio', torch.zeros(
                emb.shape[0], emb.shape[1], self.proprio_dim, device=emb.device
            )),
        }
        return result

    @torch.no_grad()
    def encode_observation_to_vector(self, obs):
        """Encode observations to a flat vector for planner initialization.

        Args:
            obs: dict with 'visual' (B, T, C, H, W) and 'proprio' (B, T, D)

        Returns:
            z: (B, T, 1, emb_dim) tensor with grad enabled
        """
        z_dict = self.encode_obs(obs)
        z = z_dict['visual']  # (B, T, 1, D)
        z = z.detach().clone()
        z.requires_grad = True
        return z

    def encode_act(self, act):
        """Encode actions through JEPA's action encoder.

        Args:
            act: (B, T, action_dim) raw actions

        Returns:
            act_emb: (B, T, emb_dim) action embeddings
        """
        return self.jepa.action_encoder(act)

    def rollout(self, obs_0, act):
        """Unroll world model from initial observation with actions.

        Args:
            obs_0: dict with 'visual' (B, T_init, C, H, W), 'proprio' (B, T_init, D)
            act: (B, T_total, action_dim) where T_total = T_init + horizon

        Returns:
            z_obses: dict with 'visual' (B, T_total+1, 1, D), 'proprio' (B, T_total+1, D)
            z: (B, T_total+1, 1, D) full embedding trajectory
        """
        pixels = self._obs_to_pixels(obs_0)
        B = pixels.shape[0]
        T_init = pixels.shape[1]
        T_total = act.shape[1]
        horizon = T_total - T_init

        # Encode initial frames
        info = {'pixels': pixels}
        info = self.jepa.encode(info)
        emb = info['emb']  # (B, T_init, D)

        # Autoregressive rollout
        all_act = act  # (B, T_total, action_dim)
        act_0 = all_act[:, :T_init]
        act_future = all_act[:, T_init:]

        # Build initial action embeddings
        act_emb_init = self.jepa.action_encoder(act_0)  # (B, T_init, D_act)

        HS = self.history_size
        current_emb = emb  # (B, T_init, D)
        current_act = act_0  # (B, T_init, action_dim)

        for t in range(horizon):
            act_emb = self.jepa.action_encoder(current_act)
            emb_trunc = current_emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred = self.jepa.predict(emb_trunc, act_trunc)[:, -1:]  # (B, 1, D)
            current_emb = torch.cat([current_emb, pred], dim=1)

            next_act = act_future[:, t:t+1]
            current_act = torch.cat([current_act, next_act], dim=1)

        # Final prediction
        act_emb = self.jepa.action_encoder(current_act)
        emb_trunc = current_emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred = self.jepa.predict(emb_trunc, act_trunc)[:, -1:]
        current_emb = torch.cat([current_emb, pred], dim=1)

        # Shape: (B, T_total+1, D) -> add patch dim
        z = current_emb.unsqueeze(2)  # (B, T_total+1, 1, D)

        proprio = obs_0.get('proprio', torch.zeros(
            B, T_init, self.proprio_dim, device=pixels.device
        ))
        # Pad proprio to match time dimension
        proprio_pad = torch.zeros(
            B, z.shape[1] - T_init, self.proprio_dim, device=pixels.device
        )
        proprio_full = torch.cat([proprio, proprio_pad], dim=1)

        z_obses = {
            'visual': z,
            'proprio': proprio_full,
        }
        return z_obses, z

    def rollout_encoded(self, z_obs_0, act):
        """Unroll the world model from already-encoded latents.

        Mirrors rollout() with the encoding step removed, so that
        rollout_encoded(encode_obs(obs)['visual'], act) reproduces
        rollout(obs, act) exactly. Used by GRASP and LatCO, which roll out
        from virtual states that never correspond to real pixels.

        Args:
            z_obs_0: (B, T_init, 1, D) encoded states, or a dict with 'visual'
            act: (B, T_total, action_dim) where T_total = T_init + horizon

        Returns:
            z_obses: dict with 'visual' (B, T_total+1, 1, D), 'proprio'
            z: (B, T_total+1, 1, D + action_dim) state embeddings with a
               zero action block appended. LeWM does not concatenate action
               embeddings into its latents (see separate_emb), while GRASP
               recovers states as z[..., :-wm.action_dim]; the zero block
               makes that slice well defined and exactly state-preserving.
        """
        if isinstance(z_obs_0, dict):
            z_obs_0 = z_obs_0['visual']

        emb = z_obs_0.squeeze(2)  # (B, T_init, D)
        B, T_init, _ = emb.shape
        T_total = act.shape[1]
        horizon = T_total - T_init

        act_0 = act[:, :T_init]
        act_future = act[:, T_init:]

        HS = self.history_size
        current_emb = emb
        current_act = act_0

        for t in range(horizon):
            act_emb = self.jepa.action_encoder(current_act)
            pred = self.jepa.predict(current_emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            current_emb = torch.cat([current_emb, pred], dim=1)
            current_act = torch.cat([current_act, act_future[:, t:t + 1]], dim=1)

        # Final prediction, matching rollout()
        act_emb = self.jepa.action_encoder(current_act)
        pred = self.jepa.predict(current_emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        current_emb = torch.cat([current_emb, pred], dim=1)

        z_state = current_emb.unsqueeze(2)  # (B, T_total+1, 1, D)
        z = torch.cat(
            [z_state, z_state.new_zeros(*z_state.shape[:-1], self.action_dim)],
            dim=-1,
        )

        proprio = z_state.new_zeros(B, z_state.shape[1], self.proprio_dim)
        z_obses = {'visual': z_state, 'proprio': proprio}
        return z_obses, z

    def next_state(self, z_state, actions):
        """Predict next states given current states and actions.

        Used by GRASP and LatCO planners for pairwise dynamics.

        Args:
            z_state: (B, T+1, 1, D) state sequence (with patch dim)
            actions: (B, T, action_dim) action sequence

        Returns:
            z_next: (B, T, 1, D) predicted next states
        """
        B, T_plus_1, P, D = z_state.shape
        T = actions.shape[1]

        # Remove patch dimension for JEPA
        emb = z_state.squeeze(2)  # (B, T+1, D)
        act_emb = self.jepa.action_encoder(actions)  # (B, T, D_act)

        HS = self.history_size

        if T <= HS:
            # Can predict all at once
            emb_input = emb[:, :T]
            pred = self.jepa.predict(emb_input, act_emb)  # (B, T, D)
        else:
            # Windowed prediction
            preds = []
            for t in range(T):
                start = max(0, t + 1 - HS)
                end = t + 1
                emb_window = emb[:, start:end]  # (B, <=HS, D)
                act_window = act_emb[:, start:end]
                p = self.jepa.predict(emb_window, act_window)[:, -1:]  # (B, 1, D)
                preds.append(p)
            pred = torch.cat(preds, dim=1)  # (B, T, D)

        return pred.unsqueeze(2)  # (B, T, 1, D)

    def predict(self, z):
        """Predict next state from encoded state (VWorldModel interface).

        Args:
            z: (B, T, 1, D) encoded states with patch dim

        Returns:
            z_pred: (B, T, 1, D) predicted states
        """
        # This method is called without explicit actions - not directly
        # compatible with JEPA which requires action conditioning.
        # For planners that call this, they should use rollout() or next_state() instead.
        raise NotImplementedError(
            "LeWM requires action-conditioned prediction. "
            "Use rollout() or next_state() instead."
        )

    def decode_obs(self, z_obs):
        """No decoder available for LeWM - return embeddings as-is."""
        return z_obs, torch.tensor(0.0)

    def visualize(self, z):
        """No visual decoder available for LeWM."""
        return None

    def separate_emb(self, z, include_actions=True):
        """Separate combined embedding into obs and action parts.

        For LeWM, there are no action embeddings concatenated into z,
        so we return z as visual and zeros as proprio.
        """
        z_obs = {
            'visual': z,
            'proprio': torch.zeros(
                z.shape[0], z.shape[1], self.proprio_dim, device=z.device
            ),
        }
        return z_obs, None

    def combine_emb(self, z_obs, z_act=None):
        """Combine observation embeddings (no-op for LeWM)."""
        if isinstance(z_obs, dict):
            return z_obs['visual']
        return z_obs

    def replace_actions_from_z(self, z, act):
        """No-op for LeWM - actions are not concatenated into z."""
        return z

    def encode(self, obs, act):
        """Combined encode matching VWorldModel signature."""
        z_dict = self.encode_obs(obs)
        return z_dict['visual']  # (B, T, 1, D)

    @property
    def action_encoder(self):
        return self.jepa.action_encoder

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        super().eval()
        self.jepa.eval()
        return self
