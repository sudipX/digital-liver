import torch
import torch.nn as nn
import torch.nn.functional as F

F_IDX, D_IDX, S_IDX, P_IDX = 0, 1, 2, 3
A_IDX, C_IDX, M_IDX, FLARE_IDX = 4, 5, 6, 7
M_UPPER = 2.0


class ConstraintAwareDecoder(nn.Module):
    """Decodes latent into next clinical state.

    Ratchet fields (F, D, P, M) use softplus-parameterised increments
    so monotonicity is guaranteed by construction, not by post-hoc clipping.
    Clipping blocks gradients when the constraint is active - softplus
    keeps gradients flowing everywhere.

    S uses a conditional branch: softplus increment in normal months,
    direct sigmoid value at ERCP months (since ERCP can decrease S).

    A, C, flare are predicted directly with sigmoid bounds - no
    monotonicity constraint on these fields.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Ratchet heads - output goes through softplus to get positive increment
        self.head_F = nn.Linear(hidden_dim, 1)
        self.head_D = nn.Linear(hidden_dim, 1)
        self.head_P = nn.Linear(hidden_dim, 1)
        self.head_M = nn.Linear(hidden_dim, 1)

        # S has two branches - selected based on ERCP flag
        self.head_S_incr  = nn.Linear(hidden_dim, 1)
        self.head_S_ercp  = nn.Linear(hidden_dim, 1)

        # Free fields - direct sigmoid output
        self.head_A = nn.Linear(hidden_dim, 1)
        self.head_C = nn.Linear(hidden_dim, 1)
        self.head_flare = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        x_prev: torch.Tensor,
        ercp_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: latent, shape (batch, latent_dim)
            x_prev: previous state x(t), shape (batch, 8)
            ercp_mask: 1.0 if ERCP this step, shape (batch,)
        Returns:
            x_next: shape (batch, 8), ratchet constraints guaranteed
        """
        h = self.trunk(z)
        batch = z.shape[0]
        x_next = torch.zeros(batch, 8, device=z.device)

        # Ratchet fields - x(t+1) = x(t) + softplus(h) always >= x(t)
        delta_F = F.softplus(self.head_F(h)).squeeze(-1)
        delta_D = F.softplus(self.head_D(h)).squeeze(-1)
        delta_P = F.softplus(self.head_P(h)).squeeze(-1)
        delta_M = F.softplus(self.head_M(h)).squeeze(-1)

        x_next[:, F_IDX] = x_prev[:, F_IDX] + delta_F
        x_next[:, D_IDX] = x_prev[:, D_IDX] + delta_D
        x_next[:, P_IDX] = x_prev[:, P_IDX] + delta_P

        # M capped at 2.0 - clamp the headroom so M never exceeds upper bound
        m_headroom = (M_UPPER - x_prev[:, M_IDX]).clamp(min=0)
        x_next[:, M_IDX] = x_prev[:, M_IDX] + torch.minimum(delta_M, m_headroom)

        # S - pick branch based on ERCP flag (hard select, not soft blend)
        s_normal = x_prev[:, S_IDX] + F.softplus(self.head_S_incr(h)).squeeze(-1)
        s_ercp   = torch.sigmoid(self.head_S_ercp(h)).squeeze(-1)
        x_next[:, S_IDX] = ercp_mask * s_ercp + (1 - ercp_mask) * s_normal

        # Free fields
        x_next[:, A_IDX] = torch.sigmoid(self.head_A(h)).squeeze(-1)
        x_next[:, C_IDX] = torch.sigmoid(self.head_C(h)).squeeze(-1)
        x_next[:, FLARE_IDX] = torch.sigmoid(self.head_flare(h)).squeeze(-1)

        return x_next


class SafetyNet(nn.Module):
    """Final deterministic layer - clips anything that somehow slipped through.

    With the softplus decoder this should never change anything.
    If it does, something is wrong upstream and worth investigating.
    """

    def forward(
        self,
        x_prev: torch.Tensor,
        x_pred: torch.Tensor,
        ercp_mask: torch.Tensor
    ) -> torch.Tensor:
        out = x_pred.clone()

        for idx in [F_IDX, D_IDX, P_IDX, M_IDX]:
            out[:, idx] = torch.maximum(x_pred[:, idx], x_prev[:, idx])

        # S only enforced in non-ERCP months
        non_ercp = (1 - ercp_mask).bool()
        s_safe = torch.where(
            non_ercp,
            torch.maximum(x_pred[:, S_IDX], x_prev[:, S_IDX]),
            x_pred[:, S_IDX]
        )
        out[:, S_IDX] = s_safe

        lower = torch.zeros(8, device=out.device)
        upper = torch.ones(8, device=out.device)
        upper[M_IDX] = M_UPPER
        out = torch.clamp(out, min=lower, max=upper)

        return out

    def count_corrections(
        self,
        x_prev: torch.Tensor,
        x_pred: torch.Tensor,
        ercp_mask: torch.Tensor
    ) -> dict:
        """How many values did the safety net actually have to change.
        Should be zero in normal operation."""
        x_safe = self.forward(x_prev, x_pred, ercp_mask)
        diffs = (x_safe - x_pred).abs()
        corrected = (diffs > 1e-6).sum(dim=0)
        names = ['F', 'D', 'S', 'P', 'A', 'C', 'M', 'flare']
        return {n: int(corrected[i].item()) for i, n in enumerate(names)}
