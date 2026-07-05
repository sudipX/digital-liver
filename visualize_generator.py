import numpy as np
import matplotlib.pyplot as plt
from generator import generate_trajectory, generate_dataset

# Field indices
F_IDX, D_IDX, S_IDX, P_IDX = 0, 1, 2, 3
A_IDX, C_IDX, M_IDX, FLARE_IDX = 4, 5, 6, 7
FIELD_NAMES = ['F', 'D', 'S', 'P', 'A', 'C', 'M', 'flare']

N_MONTHS = 48  # longer trajectory gives clearer patterns


# Plot 1: Ratchet fields over time for 10 patients
# Expected: F, D, P, M all monotonically increasing.
# High-susceptibility patients rise faster than low-susceptibility ones.
def plot_ratchet_fields():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Plot 1: Ratchet Fields (10 patients)\nShould all be monotonically increasing", fontsize=13)

    ratchet_fields = [(F_IDX, 'F - Fibrosis'), (D_IDX, 'D - Ductopenia'),
                      (P_IDX, 'P - Portal Hypertension'), (M_IDX, 'M - Malignancy Hazard')]

    for ax, (idx, title) in zip(axes.flat, ratchet_fields):
        for seed in range(10):
            traj, ctx = generate_trajectory(
                N_MONTHS, disease_class=0, age_normalized=0.5,
                sex=0, responder=0, udca_start_month=999,
                ercp_months=[], seed=seed
            )
            # Color by susceptibility - darker means higher susceptibility
            color = plt.cm.Blues(0.3 + 0.6 * ctx['susceptibility'])
            ax.plot(traj[:, idx], color=color, alpha=0.8, linewidth=1.2)

        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")
        if idx == M_IDX:
            ax.set_ylim(0, 2.0)
        else:
            ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.3, 1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.flat[-1], label='Susceptibility')

    plt.tight_layout()
    plt.savefig('plot1_ratchet_fields.png', dpi=120, bbox_inches='tight')
    print("Saved plot1_ratchet_fields.png")
    plt.close()


# Plot 2: A and C for responder vs non-responder
# Expected: after UDCA start month, responder A and C should clearly drop
# and stay lower than non-responder.
def plot_responder_vs_nonresponder():
    udca_start = 12

    traj_resp, _ = generate_trajectory(
        N_MONTHS, disease_class=0, age_normalized=0.5,
        sex=0, responder=1, udca_start_month=udca_start,
        ercp_months=[], seed=42
    )
    traj_non, _ = generate_trajectory(
        N_MONTHS, disease_class=0, age_normalized=0.5,
        sex=0, responder=0, udca_start_month=udca_start,
        ercp_months=[], seed=42  # same seed = same patient except responder status
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Plot 2: Responder vs Non-Responder (UDCA starts month {udca_start})\n"
        "After the dashed line, responder A and C should be visibly lower",
        fontsize=13
    )

    for ax, idx, label in zip(axes, [A_IDX, C_IDX], ['A - Inflammatory Activity', 'C - Cholestasis']):
        ax.plot(traj_resp[:, idx], color='steelblue', label='Responder', linewidth=1.5)
        ax.plot(traj_non[:, idx],  color='tomato',    label='Non-responder', linewidth=1.5)
        ax.axvline(udca_start, color='gray', linestyle='--', linewidth=1.5, label=f'UDCA start (month {udca_start})')
        ax.set_title(label)
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot2_responder_vs_nonresponder.png', dpi=120, bbox_inches='tight')
    print("Saved plot2_responder_vs_nonresponder.png")
    plt.close()


# Plot 3: S field for a patient with ERCP at months 12 and 24
# Expected: S increases gradually, drops at month 12, increases again,
# drops at month 24. Step-function pattern.
def plot_ercp_effect():
    ercp_months = [12, 24]

    traj_ercp, _ = generate_trajectory(
        N_MONTHS, disease_class=0, age_normalized=0.5,
        sex=0, responder=0, udca_start_month=999,
        ercp_months=ercp_months, seed=7
    )
    traj_no_ercp, _ = generate_trajectory(
        N_MONTHS, disease_class=0, age_normalized=0.5,
        sex=0, responder=0, udca_start_month=999,
        ercp_months=[], seed=7  # same patient without ERCP
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Plot 3: S - Biliary Strictures with ERCP at months 12 and 24\n"
        "Should show gradual increase with sharp drops at ERCP months",
        fontsize=13
    )

    ax.plot(traj_ercp[:, S_IDX],    color='steelblue', label='With ERCP', linewidth=2)
    ax.plot(traj_no_ercp[:, S_IDX], color='lightgray', label='No ERCP',   linewidth=1.5, linestyle='--')

    for m in ercp_months:
        ax.axvline(m, color='tomato', linestyle='--', linewidth=1.5, label=f'ERCP month {m}')

    ax.set_xlabel("Month")
    ax.set_ylabel("S value")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot3_ercp_effect.png', dpi=120, bbox_inches='tight')
    print("Saved plot3_ercp_effect.png")
    plt.close()


# Plot 4: Flare events - flare field, A, and C
# Expected: flare spikes suddenly then decays over ~6 months.
# At each flare spike, A and C should also show simultaneous smaller spikes.
def plot_flare_events():
    # Use a seed likely to have a few flares in 48 months
    # With 5% monthly probability, 48 months gives ~2.4 expected flares
    # Try a few seeds until we find one with visible flares
    traj = None
    for seed in range(100):
        t, ctx = generate_trajectory(
            N_MONTHS, disease_class=0, age_normalized=0.5,
            sex=0, responder=0, udca_start_month=999,
            ercp_months=[], seed=seed
        )
        # Check if there are any significant flare spikes
        if t[:, FLARE_IDX].max() > 0.5:
            traj = t
            used_seed = seed
            break

    if traj is None:
        print("No flare-rich trajectory found in first 100 seeds, using seed 0")
        traj, _ = generate_trajectory(N_MONTHS, 0, 0.5, 0, 0, 999, [], 0)
        used_seed = 0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Plot 4: Flare Events (seed={used_seed})\n"
        "Flare spikes then decays. A and C should show simultaneous smaller spikes.",
        fontsize=13
    )

    months = np.arange(N_MONTHS)

    axes[0].plot(months, traj[:, FLARE_IDX], color='tomato', linewidth=1.5)
    axes[0].set_ylabel("Flare")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Flare field - sharp spike then exponential decay")

    axes[1].plot(months, traj[:, A_IDX], color='steelblue', linewidth=1.5)
    axes[1].set_ylabel("A (Inflammation)")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("A - should spike simultaneously with flare")

    axes[2].plot(months, traj[:, C_IDX], color='darkorange', linewidth=1.5)
    axes[2].set_ylabel("C (Cholestasis)")
    axes[2].set_xlabel("Month")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("C - should spike simultaneously with flare")

    plt.tight_layout()
    plt.savefig('plot4_flare_events.png', dpi=120, bbox_inches='tight')
    print(f"Saved plot4_flare_events.png  (seed={used_seed})")
    plt.close()


# Plot 5: M accumulation rate vs F*C product
# Expected: delta_M(t) = M(t+1) - M(t) should track F(t)*C(t) closely.
# If the correlation is not visible, there is a bug in the generator.
def plot_m_accumulation():
    traj, ctx = generate_trajectory(
        N_MONTHS, disease_class=0, age_normalized=0.5,
        sex=0, responder=0, udca_start_month=999,
        ercp_months=[], seed=3
    )

    months  = np.arange(N_MONTHS - 1)
    delta_M = traj[1:, M_IDX] - traj[:-1, M_IDX]   # M(t+1) - M(t)
    fc_prod = traj[:-1, F_IDX] * traj[:-1, C_IDX]   # F(t) * C(t)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(
        "Plot 5: M Accumulation Rate vs F*C Product\n"
        "delta_M should track F*C. If not correlated, there is a bug.",
        fontsize=13
    )

    # Time series comparison
    ax1 = axes[0]
    ax1.plot(months, delta_M, color='purple', label='delta_M(t)', linewidth=1.5)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(months, fc_prod, color='darkorange', label='F(t)*C(t)', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("delta_M", color='purple')
    ax1_twin.set_ylabel("F*C", color='darkorange')
    ax1.set_title("Over time - both lines should move together")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Scatter plot - should look linear if generator is correct
    ax2 = axes[1]
    ax2.scatter(fc_prod, delta_M, alpha=0.5, s=20, color='steelblue')
    ax2.set_xlabel("F(t) * C(t)")
    ax2.set_ylabel("delta_M(t)")
    ax2.set_title("Scatter: should look like a line through the origin")
    ax2.grid(True, alpha=0.3)

    # Add correlation value
    corr = np.corrcoef(fc_prod, delta_M)[0, 1]
    ax2.annotate(
        f"Pearson r = {corr:.3f}",
        xy=(0.05, 0.90), xycoords='axes fraction',
        fontsize=11, color='tomato'
    )

    plt.tight_layout()
    plt.savefig('plot5_m_accumulation.png', dpi=120, bbox_inches='tight')
    print(f"Saved plot5_m_accumulation.png  (Pearson r = {corr:.3f})")
    plt.close()


# Run all plots
if __name__ == '__main__':
    print("Generating visualizations...")
    print("(Comment out this file before training)\n")

    plot_ratchet_fields()
    plot_responder_vs_nonresponder()
    plot_ercp_effect()
    plot_flare_events()
    plot_m_accumulation()

    print("\nAll plots saved. Review them before writing any model code.")
    print("What to check:")
    print("  plot1 - all ratchet fields should be monotone increasing")
    print("  plot2 - responder A and C should drop clearly after UDCA start")
    print("  plot3 - S should drop sharply at ERCP months then creep back up")
    print("  plot4 - flare spike and simultaneous A/C spikes should be visible")
    print("  plot5 - Pearson r between delta_M and F*C should be close to 1.0")