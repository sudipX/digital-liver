import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from model import LiverWorldModel, encode_context
from anticollapse import CollapseMonitor
from generator import (
    generate_dataset,
    generate_controlled_susceptibility,
    verify_generator,
    M_ACCUMULATION_RATE,
)

F_IDX, D_IDX, S_IDX, P_IDX = 0, 1, 2, 3
A_IDX, C_IDX, M_IDX, FLARE_IDX = 4, 5, 6, 7
FIELD_NAMES = ['F', 'D', 'S', 'P', 'A', 'C', 'M', 'flare']


def constraint_violation_rate(
    model: LiverWorldModel,
    trajs: List[np.ndarray],
    ctxs: List[dict],
    device: torch.device,
) -> dict:
    """Measures violations before and after the safety net.

    With softplus decoder, pre-safety-net should already be 0.
    If it is not, the decoder has a bug. Post-safety-net being 0
    is the hard guarantee from the projection layer.
    """
    model.eval()
    pre  = {f: 0 for f in ['F', 'D', 'P', 'M', 'S_no_ercp']}
    post = {f: 0 for f in ['F', 'D', 'P', 'M', 'S_no_ercp']}
    total = 0

    with torch.no_grad():
        for traj, ctx in zip(trajs, ctxs):
            ercp_set = set(ctx['ercp_months'])
            for t in range(len(traj) - 1):
                xc   = torch.tensor(traj[t],   dtype=torch.float32).unsqueeze(0).to(device)
                xn   = torch.tensor(traj[t+1], dtype=torch.float32).unsqueeze(0).to(device)
                ctxc = encode_context(ctx, t).unsqueeze(0).to(device)
                ctxn = encode_context(ctx, t+1).unsqueeze(0).to(device)
                is_ercp = (t + 1) in ercp_set
                ercp = torch.tensor([float(is_ercp)], dtype=torch.float32).to(device)

                out = model(xc, ctxc, ercp, xn, ctxn)
                raw   = out['x_pred'].squeeze(0).cpu().numpy()
                final = out['x_final'].squeeze(0).cpu().numpy()
                prev = xc.squeeze(0).cpu().numpy() 
                total += 1

                for field, idx in [('F', F_IDX), ('D', D_IDX), ('P', P_IDX), ('M', M_IDX)]:
                    if raw[idx]   < prev[idx] - 1e-5: pre[field]  += 1
                    if final[idx] < prev[idx] - 1e-5: post[field] += 1

                if not is_ercp:
                    if raw[S_IDX]   < prev[S_IDX] - 1e-5: pre['S_no_ercp']  += 1
                    if final[S_IDX] < prev[S_IDX] - 1e-5: post['S_no_ercp'] += 1

    print(f"\nConstraint violations ({total} transitions)")
    print(f"{'Field':<14}{'Pre-safety-net':>16}{'Post-safety-net':>16}")
    print("-" * 46)
    for f in pre:
        print(f"  {f:<12}{pre[f]/total*100:>14.4f}%{post[f]/total*100:>14.4f}%")

    return {'pre': pre, 'post': post, 'total': total}


def predictive_accuracy(
    model: LiverWorldModel,
    trajs: List[np.ndarray],
    ctxs: List[dict],
    history_months: int = 12,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """Rolling forecast: give model first history_months, predict the rest."""
    model.eval()
    all_mae = []
    field_mae = {f: [] for f in range(8)}

    with torch.no_grad():
        for traj, ctx in zip(trajs, ctxs):
            n_pred = len(traj) - history_months
            if n_pred <= 0:
                continue
            # rollout runs on cpu internally
            preds = model.rollout(traj[:history_months], ctx, n_pred)
            truth = traj[history_months:]
            all_mae.append(np.mean(np.abs(preds - truth)))
            for f in range(8):
                field_mae[f].append(np.mean(np.abs(preds[:, f] - truth[:, f])))

    result = {
        'mean': float(np.mean(all_mae)),
        'std':  float(np.std(all_mae)),
        'per_field': {FIELD_NAMES[f]: float(np.mean(field_mae[f])) for f in range(8)},
    }
    print(f"\nPredictive accuracy (history={history_months}mo):")
    print(f"  Overall MAE: {result['mean']:.4f} +/- {result['std']:.4f}")
    for name, val in result['per_field'].items():
        print(f"  {name:<8}: {val:.4f}")
    return result


def coupling_check(
    model: LiverWorldModel,
    trajs: List[np.ndarray],
    ctxs: List[dict],
    device: torch.device,
) -> dict:
    """Checks that M's predicted increment tracks the F*C product.

    This is separate from the monotonicity check - a model can satisfy
    M non-decreasing while still accumulating at the wrong rate.
    Uses the generator's true rate for comparison (eval-only, never given to model).
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for traj, ctx in zip(trajs, ctxs):
            for t in range(len(traj) - 1):
                xc   = torch.tensor(traj[t], dtype=torch.float32).unsqueeze(0).to(device)
                ctxc = encode_context(ctx, t).unsqueeze(0).to(device)
                ercp = torch.tensor(
                    [float((t+1) in ctx['ercp_months'])], dtype=torch.float32
                ).to(device)

                z  = model.encoder(xc, ctxc)
                zp = model.predictor(z)
                xp = model.decoder(zp, xc, ercp)
                xf = model.safety(xc, xp, ercp)

                pred_dM = (xf[0, M_IDX] - xc[0, M_IDX]).item()
                # True rate uses hidden susceptibility - only valid for eval
                true_dM = traj[t, F_IDX] * traj[t, C_IDX] * M_ACCUMULATION_RATE * ctx['susceptibility']
                errors.append(abs(pred_dM - true_dM))

    mean_err = float(np.mean(errors))
    print(f"\nCoupling check (M increment vs true F*C rate):")
    print(f" Mean absolute error: {mean_err:.6f}")
    print(f" Model learned log_m_rate -> rate = {np.exp(model.log_m_rate.item()):.6f}")
    print(f" Generator true rate = {M_ACCUMULATION_RATE:.6f}")
    return {'mean_error': mean_err}


def generalisation_probes(
    model: LiverWorldModel,
    device: torch.device,
    n_months: int = 24,
) -> dict:
    """Three probes designed to find where the model breaks."""
    results = {}

    # Baseline - in-distribution
    print("\n[Baseline] In-distribution test set")
    base_t, base_c = generate_dataset(100, n_months, seed_offset=5000)
    results['baseline'] = predictive_accuracy(model, base_t, base_c, device=device)
    base_mae = results['baseline']['mean']

    # Probe 1 - high susceptibility patients
    print("\n[Probe 1] High susceptibility (0.80-1.00 vs train 0.30-1.00)")
    p1_t, p1_c = generate_controlled_susceptibility(100, n_months, 0.80, 1.00, seed_offset=20000)
    results['probe1_high_susc'] = predictive_accuracy(model, p1_t, p1_c, device=device)

    # Probe 2 - late treatment start
    print("\n[Probe 2] Late UDCA start (months 13-18 vs train months 2-8)")
    p2_t, p2_c = generate_dataset(100, n_months, seed_offset=30000, udca_start_range=(13, 19))
    results['probe2_late_treatment'] = predictive_accuracy(model, p2_t, p2_c, device=device)

    # Probe 3 - longer rollout than training
    print("\n[Probe 3] 48-month rollout (trained on 24 months)")
    p3_t, p3_c = generate_dataset(50, 48, seed_offset=40000)
    results['probe3_long_rollout'] = predictive_accuracy(model, p3_t, p3_c, device=device)

    print("\nGeneralisation summary (vs baseline MAE={:.4f}):".format(base_mae))
    for key in ['probe1_high_susc', 'probe2_late_treatment', 'probe3_long_rollout']:
        mae = results[key]['mean']
        deg = (mae - base_mae) / base_mae * 100
        print(f"  {key:<30}: MAE={mae:.4f}  ({deg:+.1f}%)")

    print("\nNote: these probes test generalisation within the generator's")
    print("distribution only. They cannot establish real-world validity.")
    return results


def explain_prediction(
    model: LiverWorldModel,
    traj: np.ndarray,
    ctx: dict,
    target_month: int = 20,
    history_months: int = 12,
) -> None:
    """Traces a specific prediction - for the memo's example trajectory."""
    model.eval()
    preds = model.rollout(traj[:history_months], ctx, target_month - history_months)
    pred  = preds[-1]

    print(f"\nExplaining prediction at month {target_month}")
    print(f"Patient: disease_class={ctx['disease_class']}, responder={ctx['responder']}, "
          f"udca_start={ctx['udca_start_month']}")
    print(f"\nPredicted state:")
    for i, name in enumerate(FIELD_NAMES):
        bar = '#' * int(pred[i] * 20)
        print(f"  {name:<8}: {pred[i]:.3f} |{bar}")

    hist_F  = traj[:history_months, F_IDX]
    hist_C  = traj[:history_months, C_IDX]
    fc_mean = (hist_F * hist_C).mean()

    print(f"\nKey history stats (months 0-{history_months-1}):")
    print(f"  Mean F: {hist_F.mean():.3f}, Mean C: {hist_C.mean():.3f}")
    print(f"  Mean F*C: {fc_mean:.4f}")
    print(f"  Predicted M at month {target_month}: {pred[M_IDX]:.3f}")

    if ctx['responder'] == 0:
        note = "Non-responder: UDCA did not suppress C, so F*C product stayed elevated."
    else:
        note = "Responder: UDCA suppressed C, slowing M accumulation after treatment."
    print(f"  Treatment note: {note}")


def run_all(model_path: str = 'best_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")

    model = LiverWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_trajs, test_ctxs = generate_dataset(200, 24, seed_offset=5000)

    print("\n" + "="*60)
    predictive_accuracy(model, test_trajs, test_ctxs, device=device)
    constraint_violation_rate(model, test_trajs, test_ctxs, device)
    coupling_check(model, test_trajs, test_ctxs, device)

    # Collapse check on sample of latents
    monitor = CollapseMonitor()
    latents = []
    with torch.no_grad():
        for traj, ctx in zip(test_trajs[:100], test_ctxs[:100]):
            for t in range(0, len(traj), 4):
                xc   = torch.tensor(traj[t], dtype=torch.float32).unsqueeze(0).to(device)
                ctxc = encode_context(ctx, t).unsqueeze(0).to(device)
                z = model.encoder(xc, ctxc)
                latents.append(z.squeeze(0).cpu())
    monitor.report(torch.stack(latents), label="test set")

    generalisation_probes(model, device)
    explain_prediction(model, test_trajs[0], test_ctxs[0])
    print("="*60)


if __name__ == '__main__':
    run_all()
