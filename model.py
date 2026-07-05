import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from constraints import ConstraintAwareDecoder, SafetyNet

F_IDX, D_IDX, S_IDX, P_IDX = 0, 1, 2, 3
A_IDX, C_IDX, M_IDX, FLARE_IDX = 4, 5, 6, 7

X_DIM       = 8
CONTEXT_DIM = 6   # disease_class, age, sex, responder, udca_active, ercp_active
LATENT_DIM  = 16
HIDDEN_DIM  = 32
N_INTERACT  = 3   # explicit coupling features: F*C, flare*A, flare*C


def encode_context(ctx: dict, month: int) -> torch.Tensor:
    """Build the 6-dim context vector for a given month."""
    return torch.tensor([
        float(ctx['disease_class']),
        float(ctx['age_normalized']),
        float(ctx['sex']),
        float(ctx['responder']),
        float(month >= ctx['udca_start_month']),
        float(month in ctx['ercp_months']),
    ], dtype=torch.float32)


def compute_interactions(x: torch.Tensor) -> torch.Tensor:
    """Explicit interaction features for the three assignment-stated couplings.

    F*C - M accumulates from this product (coupling 3)
    flare*A - flare perturbs A (coupling 1)
    flare*C - flare perturbs C (coupling 1)

    These are hand-specified because the assignment states them explicitly.
    No other pairwise products are added to avoid inventing couplings.
    """
    return torch.stack([
        x[:, F_IDX] * x[:, C_IDX],
        x[:, FLARE_IDX] * x[:, A_IDX],
        x[:, FLARE_IDX] * x[:, C_IDX],
    ], dim=-1)


class Encoder(nn.Module):
    """Encodes [x(t), context, interaction_features] -> z(t).

    Input is augmented with known interaction terms so the network
    doesn't have to rediscover multiplicative relationships from scratch.
    """

    def __init__(self):
        super().__init__()
        input_dim = X_DIM + CONTEXT_DIM + N_INTERACT
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        interactions = compute_interactions(x)
        inp = torch.cat([x, ctx, interactions], dim=-1)
        return self.net(inp)


class Predictor(nn.Module):
    """Predicts z(t+1) from z(t) - the JEPA core step.

    Does not receive context directly. The encoder already folded
    context into z(t), so the predictor works in pure latent space.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LiverWorldModel(nn.Module):
    """JEPA-style world model: encoder -> predictor -> decoder -> safety net.

    Also owns log_m_rate, a single learned scalar for the F*C -> M
    coupling loss. Stored in log-space so it stays positive. The model
    learns this from data rather than being given the generator's rate,
    which would be leaking the answer.
    """

    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.predictor = Predictor()
        self.decoder   = ConstraintAwareDecoder()
        self.safety    = SafetyNet()

        # Start with a small uninformed guess - model must learn the right value
        self.log_m_rate = nn.Parameter(torch.tensor(float(np.log(0.001))))

    def forward(
        self,
        x_curr: torch.Tensor,
        ctx_curr: torch.Tensor,
        ercp_mask: torch.Tensor,
        x_next: Optional[torch.Tensor] = None,
        ctx_next: Optional[torch.Tensor] = None,
    ) -> dict:
        """One-step forward pass.

        x_next and ctx_next are only needed during training to compute
        the target latent (with stop-gradient).
        """
        z_curr  = self.encoder(x_curr, ctx_curr)
        z_pred  = self.predictor(z_curr)
        x_pred  = self.decoder(z_pred, x_curr, ercp_mask)
        x_final = self.safety(x_curr, x_pred, ercp_mask)

        out = {
            'z_curr':  z_curr,
            'z_pred':  z_pred,
            'x_pred':  x_pred,
            'x_final': x_final,
        }

        if x_next is not None and ctx_next is not None:
            # Stop-gradient on target encoder - prevents collapse by breaking the symmetric gradient path
            with torch.no_grad():
                z_target = self.encoder(x_next, ctx_next)
            out['z_target'] = z_target

        return out

    def rollout(
        self,
        x_history: np.ndarray,
        ctx: dict,
        n_steps: int,
    ) -> np.ndarray:
        """Autoregressive rollout from the last observed state.

        The model was trained with teacher forcing so this free-running
        mode will accumulate errors over long horizons - that's expected
        and measured in the evaluation.
        """
        self.eval()
        preds = []
        x_curr = torch.tensor(x_history[-1], dtype=torch.float32).unsqueeze(0)
        history_len = len(x_history)

        with torch.no_grad():
            for step in range(n_steps):
                month = history_len + step
                ctx_t = encode_context(ctx, month).unsqueeze(0)
                ercp  = torch.tensor(
                    [float((month + 1) in ctx['ercp_months'])],
                    dtype=torch.float32
                )
                out = self.forward(x_curr, ctx_t, ercp)
                x_curr = out['x_final']
                preds.append(x_curr.squeeze(0).numpy())

        return np.array(preds)
