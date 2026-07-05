"""
plots.py - the three figures for the memo.

Run this AFTER train.py (needs training_history.json + best_model.pt)
and independently of evaluate.py (this script re-runs the small amount
of inference it needs directly, so evaluate.py does not need to be run
first - though it's a good sanity check if the numbers should match).

Usage:
    python plots.py
Outputs:
    fig1_training_curve.png
    fig2_rollout_example.png
    fig3_generalisation_probes.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from model import LiverWorldModel, encode_context
from generator import generate_dataset
from evaluate import generalisation_probes

F_IDX, D_IDX, S_IDX, P_IDX = 0, 1, 2, 3
A_IDX, C_IDX, M_IDX, FLARE_IDX = 4, 5, 6, 7
FIELD_NAMES = ['F', 'D', 'S', 'P', 'A', 'C', 'M', 'flare']


# ---------------------------------------------------------------------
# Plot 1: training curve - val_mae + effective rank overlaid
# ---------------------------------------------------------------------
def plot_training_curve(history_path: str = 'training_history.json',
                         out_path: str = 'fig1_training_curve.png'):
    with open(history_path) as f:
        history = json.load(f)

    epochs   = [h['epoch'] for h in history]
    val_mae  = [h['val_mae'] for h in history]
    eff_rank = [h['effective_rank'] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.suptitle("Training: predictive accuracy vs. latent effective rank",
                 fontsize=13)

    color1 = 'steelblue'
    ax1.plot(epochs, val_mae, color=color1, linewidth=2, label='Val MAE')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MAE", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tomato'
    ax2.plot(epochs, eff_rank, color=color2, linewidth=2,
             linestyle='--', label='Effective rank')
    ax2.set_ylabel("Effective rank of latent (out of 16)", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 16)

    # Annotate the collapse -> recovery -> plateau phases
    ax2.axhline(16 * 0.4, color='gray', linestyle=':', linewidth=1)
    ax2.annotate("redundancy threshold (0.4 x dim)", xy=(epochs[-1], 16 * 0.4),
                 xytext=(-5, 5), textcoords='offset points',
                 ha='right', fontsize=8, color='gray')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Plot 2: rollout example - predicted vs true, for the same patient
# used in evaluate.py's explain_prediction (test_trajs[0])
# ---------------------------------------------------------------------
def plot_rollout_example(model_path: str = 'best_model.pt',
                          patient_idx: int = 0,
                          history_months: int = 12,
                          n_months: int = 24,
                          out_path: str = 'fig2_rollout_example.png'):
    device = torch.device('cpu')
    model = LiverWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Same test set / same patient as evaluate.py's explain_prediction
    test_trajs, test_ctxs = generate_dataset(200, n_months, seed_offset=5000)
    traj = test_trajs[patient_idx]
    ctx  = test_ctxs[patient_idx]

    n_pred = n_months - history_months
    preds = model.rollout(traj[:history_months], ctx, n_pred)  # (n_pred, 8)

    months_true = np.arange(n_months)
    months_pred = np.arange(history_months, n_months)

    fields_to_plot = [('F', F_IDX), ('M', M_IDX), ('A', A_IDX), ('C', C_IDX)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Rollout: patient {patient_idx} (disease_class={ctx['disease_class']}, "
        f"responder={ctx['responder']}, UDCA start=month {ctx['udca_start_month']})\n"
        f"Solid = ground truth | Dashed = model rollout from month {history_months}",
        fontsize=12
    )

    for ax, (name, idx) in zip(axes.flat, fields_to_plot):
        ax.plot(months_true, traj[:, idx], color='steelblue',
                linewidth=2, label='True')
        ax.plot(months_pred, preds[:, idx], color='tomato',
                linewidth=2, linestyle='--', label='Predicted')

        ax.axvline(history_months, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Prediction starts (month {history_months})')
        ax.axvline(ctx['udca_start_month'], color='seagreen', linestyle='-.',
                   linewidth=1.5, label=f"UDCA start (month {ctx['udca_start_month']})")
        for m in ctx['ercp_months']:
            ax.axvline(m, color='darkorange', linestyle='-.', linewidth=1.5,
                       label=f'ERCP (month {m})')

        ax.set_title(name)
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")
        if idx == M_IDX:
            ax.set_ylim(0, max(0.05, traj[:, idx].max() * 1.5))
        else:
            ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Plot 3: generalisation probes - bar chart of MAE vs baseline
# ---------------------------------------------------------------------
def plot_generalisation_probes(model_path: str = 'best_model.pt',
                                out_path: str = 'fig3_generalisation_probes.png'):
    device = torch.device('cpu')
    model = LiverWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = generalisation_probes(model, device)

    labels = ['Baseline\n(in-dist)', 'Probe 1\n(high suscept.)',
              'Probe 2\n(late UDCA)', 'Probe 3\n(48mo rollout)']
    keys = ['baseline', 'probe1_high_susc', 'probe2_late_treatment', 'probe3_long_rollout']
    maes = [results[k]['mean'] for k in keys]
    base_mae = maes[0]
    colors = ['steelblue', 'tomato', 'tomato', 'tomato']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, maes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.axhline(base_mae, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline MAE ({base_mae:.4f})')

    for bar, mae in zip(bars, maes):
        deg = (mae - base_mae) / base_mae * 100
        label = f"{mae:.4f}" if bar is bars[0] else f"{mae:.4f}\n({deg:+.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                label, ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Overall MAE (history=12mo rollout)")
    ax.set_title("Generalisation probes vs. in-distribution baseline\n"
                 "(tests generalisation within this generator only - see note)",
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    plot_training_curve()
    plot_rollout_example()
    plot_generalisation_probes()
    print("\nAll figures saved. Ready to drop into the memo.")