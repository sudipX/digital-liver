import numpy as np
from typing import List, Tuple, Dict, Optional

# Field indices
F_IDX = 0
D_IDX = 1
S_IDX = 2
P_IDX = 3
A_IDX = 4
C_IDX = 5
M_IDX = 6
FLARE_IDX = 7

# Flare parameters - my own choices, not from assignment
FLARE_PROBABILITY = 0.05
FLARE_DECAY = 0.65
FLARE_MAGNITUDE = 0.85

# A and C use the same flare effect magnitude since assignment
# says "perturb A and C together" with no distinction between them
FLARE_EFFECT = FLARE_MAGNITUDE

# Mean reversion for A and C - same params for both fields
# assignment doesn't distinguish them quantitatively
MEAN_AC = 0.20
REVERSION_SPEED = 0.10
NOISE_STD = 0.02

# Single shared ratchet rate for F, D, P
# assignment groups them together with no relative speed ordering
RATCHET_RATE = 0.012

# UDCA suppression - same factor for both A and C
UDCA_SUPPRESSION = 0.90

# M accumulation rate - my placeholder, not given by assignment
# model must learn its own estimate via log_m_rate parameter
M_ACCUMULATION_RATE = 0.006

# Stricture creep constants
S_CREEP_BASE = 0.005
S_CREEP_NOISE = 0.005


def initialize_patient(seed: int) -> np.ndarray:
    """Sample starting state x(0). Disease class does not affect
    generator dynamics - it is passed as context to the model only."""
    rng = np.random.RandomState(seed)
    x0 = np.zeros(8)
    x0[F_IDX] = rng.uniform(0.02, 0.20)
    x0[D_IDX] = rng.uniform(0.01, 0.10)
    x0[S_IDX] = rng.uniform(0.02, 0.14)
    x0[P_IDX] = rng.uniform(0.01, 0.10)
    x0[A_IDX] = rng.uniform(0.10, 0.30)
    x0[C_IDX] = rng.uniform(0.10, 0.30)
    x0[M_IDX] = 0.0
    x0[FLARE_IDX] = 0.0
    return x0


def compute_drive(A: float, C: float, susceptibility: float) -> float:
    """Shared monthly drive for F, D, P.
    Equal weighting of A and C - assignment gives no formula so
    this is the simplest reading of 'driven by A and C'."""
    return max(0.0, susceptibility * (A + C) * RATCHET_RATE)


def update_state(
    x: np.ndarray,
    susceptibility: float,
    responder: int,
    udca_active: bool,
    ercp_this_month: bool,
    rng: np.random.RandomState
) -> np.ndarray:
    """One-month state update implementing the three assignment-stated
    coupling relationships only:
      1. flares perturb A and C together
      2. treatment suppresses A and C for responders
      3. M accumulates as hazard of F*C
    """
    x_new = x.copy()
    F, D, S, P, A, C, M, flare = x

    # New flare event this month
    new_flare = bool(rng.uniform() < FLARE_PROBABILITY)

    # Flare field - exponential decay plus new spike if flare occurs
    x_new[FLARE_IDX] = flare * FLARE_DECAY
    if new_flare:
        x_new[FLARE_IDX] += FLARE_MAGNITUDE

    # Ratchet fields - same drive for all three
    drive = compute_drive(A, C, susceptibility)
    x_new[F_IDX] = F + drive
    x_new[D_IDX] = D + drive
    x_new[P_IDX] = P + drive

    # S - non-decreasing unless ERCP drops it
    if ercp_this_month:
        x_new[S_IDX] = S - rng.uniform(0.08, 0.28)
    else:
        x_new[S_IDX] = S + S_CREEP_BASE + rng.normal(0, S_CREEP_NOISE)

    # A - mean reverting, spiked by flares (coupling 1)
    x_new[A_IDX] = (
        A
        + REVERSION_SPEED * (MEAN_AC - A)
        + rng.normal(0, NOISE_STD)
        + FLARE_EFFECT * float(new_flare)
    )

    # C - same structure as A (coupling 1 applies equally)
    x_new[C_IDX] = (
        C
        + REVERSION_SPEED * (MEAN_AC - C)
        + rng.normal(0, NOISE_STD)
        + FLARE_EFFECT * float(new_flare)
    )

    # Treatment suppression for responders (coupling 2)
    if udca_active and responder == 1:
        x_new[A_IDX] *= UDCA_SUPPRESSION
        x_new[C_IDX] *= UDCA_SUPPRESSION

    # M accumulates from F*C product (coupling 3)
    # Uses current F and C (before update) - correct causal order
    x_new[M_IDX] = M + F * C * M_ACCUMULATION_RATE * susceptibility

    # Hard bounds
    for idx in [F_IDX, D_IDX, S_IDX, P_IDX, A_IDX, C_IDX, FLARE_IDX]:
        x_new[idx] = np.clip(x_new[idx], 0.0, 1.0)
    x_new[M_IDX] = np.clip(x_new[M_IDX], 0.0, 2.0)

    # Monotonicity safety net
    x_new[F_IDX] = max(x_new[F_IDX], x[F_IDX])
    x_new[D_IDX] = max(x_new[D_IDX], x[D_IDX])
    x_new[P_IDX] = max(x_new[P_IDX], x[P_IDX])
    x_new[M_IDX] = max(x_new[M_IDX], x[M_IDX])
    if not ercp_this_month:
        x_new[S_IDX] = max(x_new[S_IDX], x[S_IDX])

    return x_new


def generate_trajectory(
    n_months: int,
    disease_class: int,
    age_normalized: float,
    sex: int,
    responder: int,
    udca_start_month: int,
    ercp_months: List[int],
    seed: int
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.RandomState(seed)
    susceptibility = rng.uniform(0.30, 1.00)

    x = initialize_patient(seed)
    trajectory = [x.copy()]

    for month in range(1, n_months):
        x = update_state(
            x, susceptibility, responder,
            month >= udca_start_month,
            month in ercp_months,
            rng
        )
        trajectory.append(x.copy())

    context = {
        'disease_class': disease_class,
        'age_normalized': age_normalized,
        'sex': sex,
        'responder': responder,
        'udca_start_month': udca_start_month,
        'ercp_months': ercp_months,
        'susceptibility': susceptibility,
        'seed': seed,
    }
    return np.array(trajectory), context


def generate_dataset(
    n_patients: int,
    n_months: int,
    seed_offset: int = 0,
    udca_start_range: Tuple[int, int] = (2, 9),
    susceptibility_range: Tuple[float, float] = (0.30, 1.00),
) -> Tuple[List[np.ndarray], List[Dict]]:
    trajectories, contexts = [], []

    for i in range(n_patients):
        seed = seed_offset + i
        rng = np.random.RandomState(seed)

        disease_class  = rng.randint(0, 2)
        age_normalized = rng.uniform(0.25, 0.75)
        sex = rng.randint(0, 2)
        responder = rng.randint(0, 2)
        udca_start = rng.randint(udca_start_range[0], udca_start_range[1])
        n_ercp = rng.choice([0, 0, 1, 1, 2], p=[0.4, 0.2, 0.2, 0.1, 0.1])
        if n_ercp > 0 and n_months > 8:
            ercp_months = sorted(rng.randint(6, n_months - 2, n_ercp).tolist())
        else:
            ercp_months = []

        traj, ctx = generate_trajectory(
            n_months, disease_class, age_normalized,
            sex, responder, udca_start, ercp_months, seed
        )
        trajectories.append(traj)
        contexts.append(ctx)

    return trajectories, contexts


def generate_controlled_susceptibility(
    n_patients: int,
    n_months: int,
    susc_min: float,
    susc_max: float,
    seed_offset: int = 0,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Used for generalisation probe - forces susceptibility into a
    specific range instead of sampling from the full [0.3, 1.0]."""
    trajectories, contexts = [], []

    for i in range(n_patients):
        seed = seed_offset + i
        rng = np.random.RandomState(seed)

        disease_class  = rng.randint(0, 2)
        age_normalized = rng.uniform(0.25, 0.75)
        sex = rng.randint(0, 2)
        responder = rng.randint(0, 2)
        udca_start = rng.randint(2, 9)
        n_ercp = rng.choice([0, 0, 1, 1, 2], p=[0.4, 0.2, 0.2, 0.1, 0.1])
        if n_ercp > 0 and n_months > 8:
            ercp_months = sorted(rng.randint(6, n_months - 2, n_ercp).tolist())
        else:
            ercp_months = []

        susceptibility = rng.uniform(susc_min, susc_max)

        x = initialize_patient(seed)
        trajectory = [x.copy()]
        for month in range(1, n_months):
            x = update_state(
                x, susceptibility, responder,
                month >= udca_start,
                month in ercp_months,
                rng
            )
            trajectory.append(x.copy())

        ctx = {
            'disease_class': disease_class,
            'age_normalized': age_normalized,
            'sex': sex,
            'responder': responder,
            'udca_start_month': udca_start,
            'ercp_months': ercp_months,
            'susceptibility': susceptibility,
            'seed': seed,
        }
        trajectories.append(np.array(trajectory))
        contexts.append(ctx)

    return trajectories, contexts


def verify_generator(trajectories: List[np.ndarray], contexts: List[Dict]) -> bool:
    """Run constraint checks on generated data. Call this before training."""
    violations = {
        'F': 0, 'D': 0, 'P': 0, 'M': 0,
        'S_no_ercp': 0, 'bounds': 0
    }
    total = 0

    for traj, ctx in zip(trajectories, contexts):
        ercp_set = set(ctx['ercp_months'])
        for t in range(len(traj) - 1):
            total += 1
            if traj[t+1, F_IDX] < traj[t, F_IDX] - 1e-9: violations['F'] += 1
            if traj[t+1, D_IDX] < traj[t, D_IDX] - 1e-9: violations['D'] += 1
            if traj[t+1, P_IDX] < traj[t, P_IDX] - 1e-9: violations['P'] += 1
            if traj[t+1, M_IDX] < traj[t, M_IDX] - 1e-9: violations['M'] += 1
            if (t+1) not in ercp_set and traj[t+1, S_IDX] < traj[t, S_IDX] - 1e-9:
                violations['S_no_ercp'] += 1
            if np.any(traj[t, :7] < -1e-9) or np.any(traj[t, :7] > 1 + 1e-9):
                violations['bounds'] += 1
            if traj[t, M_IDX] > 2.0 + 1e-9:
                violations['bounds'] += 1

    all_ok = all(v == 0 for v in violations.values())
    print(f"Generator check - {total} transitions, violations: {violations}")
    print(f"All constraints satisfied: {all_ok}")
    return all_ok
