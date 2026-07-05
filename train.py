import json
import torch
import torch.nn as nn
import numpy as np

from model import LiverWorldModel, encode_context
from anticollapse import CollapseMonitor
from generator import generate_dataset, verify_generator

F_IDX, D_IDX, S_IDX, P_IDX = 0, 1, 2, 3
A_IDX, C_IDX, M_IDX, FLARE_IDX = 4, 5, 6, 7


def coupling_loss(
    x_prev: torch.Tensor,
    x_final: torch.Tensor,
    log_m_rate: torch.Tensor,
) -> torch.Tensor:
    """Penalises M increment for deviating from the F*C pattern.

    Rate is a learned parameter on the model, not copied from the generator.
    Susceptibility is hidden so this target is approximate - kept at low
    weight in the total loss.
    """
    dM_pred   = x_final[:, M_IDX] - x_prev[:, M_IDX]
    dM_target = x_prev[:, F_IDX] * x_prev[:, C_IDX] * torch.exp(log_m_rate)
    return nn.functional.mse_loss(dM_pred, dM_target)


def total_loss(
    out: dict,
    x_target: torch.Tensor,
    x_prev: torch.Tensor,
    log_m_rate: torch.Tensor,
    lam_recon: float = 1.0,
    lam_coupling: float = 0.3,
) -> tuple:
    """Two-component loss:
      1. latent_loss  - core JEPA objective, stop-gradient on target
      2. recon_loss   - decoded output vs ground truth (L1)
      3. coupling_loss - M increment vs F*C product (weak auxiliary)

    No explicit anti-collapse terms. Stop-gradient in model.forward()
    is the sole collapse-prevention mechanism. This is sufficient for
    this problem scale (8 dims, 500 patients, 16-dim latent).
    """
    l_latent = nn.functional.mse_loss(out['z_pred'], out['z_target'])
    l_recon = nn.functional.l1_loss(out['x_final'], x_target)
    l_coupling = coupling_loss(x_prev, out['x_final'], log_m_rate)

    total = l_latent + lam_recon * l_recon + lam_coupling * l_coupling

    components = {
        'latent': l_latent.item(),
        'recon': l_recon.item(),
        'coupling': l_coupling.item(),
        'total': total.item(),
    }
    return total, components


def tf_probability(epoch: int, start: float = 1.0,
                   decay: float = 0.97, floor: float = 0.3) -> float:
    """Exponential decay schedule for teacher-forcing probability."""
    return max(floor, start * (decay ** epoch))


def train(
    n_train: int = 500,
    n_val: int = 100,
    n_months: int = 24,
    n_epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    window: int = 3,
    save_path: str = 'best_model.pt',
    device_str: str = 'auto',
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    print("Generating data...")
    train_trajs, train_ctxs = generate_dataset(n_train, n_months, seed_offset=0)
    val_trajs, val_ctxs = generate_dataset(n_val,   n_months, seed_offset=10000)

    ok = verify_generator(train_trajs, train_ctxs)
    if not ok:
        raise RuntimeError("Generator constraint check failed - fix before training")

    model = LiverWorldModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    monitor = CollapseMonitor()

    best_val = float('inf')
    history = []

    print(f"Training {n_epochs} epochs | {n_train} patients | window={window}")

    for epoch in range(n_epochs):
        model.train()
        tf_prob  = tf_probability(epoch)
        ep_comps = {}
        ep_steps = 0
        total_steps_epoch = 0

        idx_perm = np.random.permutation(len(train_trajs))

        for b_start in range(0, len(idx_perm), batch_size):
            batch_idx = idx_perm[b_start: b_start + batch_size]
            b_loss = 0.0
            b_steps = 0

            for i in batch_idx:
                traj = train_trajs[i]
                ctx = train_ctxs[i]
                n = len(traj)

                max_start = max(0, n - window - 1)
                t0 = np.random.randint(0, max_start + 1)

                x_in = torch.tensor(traj[t0], dtype=torch.float32).unsqueeze(0).to(device)

                for step in range(window):
                    t  = t0 + step
                    t1 = t + 1
                    if t1 >= n:
                        break

                    ctx_t  = encode_context(ctx, t).unsqueeze(0).to(device)
                    ctx_t1 = encode_context(ctx, t1).unsqueeze(0).to(device)
                    ercp = torch.tensor(
                        [float(t1 in ctx['ercp_months'])], dtype=torch.float32
                    ).to(device)
                    x_true_next = torch.tensor(
                        traj[t1], dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    out = model(x_in, ctx_t, ercp, x_true_next, ctx_t1)
                    loss, comps = total_loss(out, x_true_next, x_in, model.log_m_rate)

                    b_loss  += loss
                    b_steps += 1

                    for k, v in comps.items():
                        ep_comps[k] = ep_comps.get(k, 0) + v

                    # Scheduled sampling
                    if np.random.uniform() < tf_prob:
                        x_in = x_true_next
                    else:
                        x_in = out['x_final'].detach()

            if b_steps > 0:
                avg = b_loss / b_steps
                opt.zero_grad()
                avg.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_steps += 1
                total_steps_epoch += b_steps

        # Validation
        model.eval()
        val_losses  = []
        val_latents = []

        with torch.no_grad():
            for traj, ctx in zip(val_trajs, val_ctxs):
                for t in range(len(traj) - 1):
                    xc = torch.tensor(traj[t],   dtype=torch.float32).unsqueeze(0).to(device)
                    xn = torch.tensor(traj[t+1], dtype=torch.float32).unsqueeze(0).to(device)
                    ctxc = encode_context(ctx, t).unsqueeze(0).to(device)
                    ctxn = encode_context(ctx, t+1).unsqueeze(0).to(device)
                    ercp = torch.tensor(
                        [float((t+1) in ctx['ercp_months'])], dtype=torch.float32
                    ).to(device)
                    out = model(xc, ctxc, ercp, xn, ctxn)
                    val_losses.append(nn.functional.l1_loss(out['x_final'], xn).item())
                    val_latents.append(out['z_pred'].squeeze(0).cpu())

        mean_val = float(np.mean(val_losses))
        sched.step(mean_val)

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), save_path)

        # Collapse diagnostic every epoch (cheap - SVD on <=300x16), printed only every 10.
        report = {'effective_rank': float('nan')}
        if val_latents:
            report = monitor.report(
                torch.stack(val_latents[:300]),
                label=f"epoch {epoch}",
                verbose=(epoch % 10 == 0 or epoch == n_epochs - 1),
            )

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            n_b = max(1, total_steps_epoch)
            print(
                f"Epoch {epoch:3d} | TF={tf_prob:.2f} | "
                f"latent={ep_comps.get('latent', 0)/n_b:.4f} | "
                f"recon={ep_comps.get('recon', 0)/n_b:.4f} | "
                f"coupling={ep_comps.get('coupling', 0)/n_b:.4f} | "
                f"val_mae={mean_val:.4f} | "
                f"log_m_rate={model.log_m_rate.item():.4f}"
            )

        history.append({
            'epoch': epoch,
            'val_mae': mean_val,
            'tf_prob': tf_prob,
            'effective_rank': report['effective_rank'],
        })

    print(f"\nDone. Best val MAE: {best_val:.4f}, saved to {save_path}")

    history_path = 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return model, history


if __name__ == '__main__':
    model, history = train()