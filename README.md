# Digital Liver World Model

A JEPA-style world model for predicting how a synthetic liver disease state evolves
month by month. Built for the Digital Liver World-Model take-home assignment.

## Table of contents

1. Overview
2. The state vector
3. Synthetic data generator
4. Model architecture
5. Constraint enforcement
6. Representation collapse: prevention and detection
7. Loss functions
8. Training procedure
9. Evaluation methodology
10. Repository structure
11. How to run
12. Design rationale

## 1. Overview

The task is to predict a patient's future clinical state $x(t+1), x(t+2), \dots$
given their history $x(0 \dots t)$ and static context (disease class, age, sex,
treatment responder status, and a treatment timeline). The approach taken is
JEPA-style: instead of predicting the raw next state directly, an encoder maps the
current state and context into a latent $z(t)$, a predictor moves that latent
forward to $z(t+1)$, and a decoder turns the predicted latent back into a clinical
state,
subject to hard constraints. A separate stop-gradient target encoder provides the
training signal for the latent prediction without letting the model collapse to a
trivial solution.

The three things this project tries to get right at once are:

- Accuracy: the predicted trajectory should track the true one.
- Constraint satisfaction: fields that can only move in one direction must never
  reverse, by construction, not by hope.
- Explainability: a prediction should be traceable back to the history and context
  that produced it.

## 2. The state vector

$x(t)$ is an 8-dimensional vector, one entry per month:

| idx | field | meaning                        | range   | temporal behaviour                              |
|-----|-------|---------------------------------|---------|--------------------------------------------------|
| 0   | F     | fibrosis                       | [0, 1]  | ratchet, non-decreasing                          |
| 1   | D     | ductopenia (duct loss)         | [0, 1]  | ratchet, non-decreasing                          |
| 2   | S     | biliary strictures             | [0, 1]  | ratchet, except may step down at an ERCP event    |
| 3   | P     | portal hypertension            | [0, 1]  | ratchet, non-decreasing                          |
| 4   | A     | inflammatory activity          | [0, 1]  | fast, mean-reverting                             |
| 5   | C     | cholestasis                    | [0, 1]  | fast, with flares                                |
| 6   | M     | malignancy hazard accumulator  | [0, 2]  | monotone non-decreasing                          |
| 7   | flare | acute cholangitis flare        | [0, 1]  | transient, decays                                |

Context supplied alongside $x(t)$, never predicted: `disease_class`, age
(normalized), sex, responder (0 or 1), UDCA start month, and a list of ERCP event
months. The 6-dimensional context vector fed to the model at month $t$ is:

$$
\text{context}(t) = \big[\, \text{disease class},\ \text{age normalized},\ \text{sex},\ \text{responder},\ \mathbb{1}\{t \geq \text{udca start month}\},\ \mathbb{1}\{t \in \text{ercp months}\} \,\big]
$$

## 3. Synthetic data generator (`generator.py`)

The generator is a seeded dynamical system. Each patient gets a hidden
susceptibility $s$, drawn once per patient from $\text{Uniform}(0.30, 1.00)$,
which scales both the ratchet drive and the M accumulation rate. Susceptibility
is never given to the model; it has to be inferred implicitly from how fast a
patient's history is moving.

### Ratchet drive (F, D, P)

All three ratchet fields share one monthly drive:

$$
\text{drive}(t) = \max\big(0,\ s \cdot (A(t) + C(t)) \cdot r\big)
$$

where $r$ is a fixed rate constant ($\text{RATCHET RATE} = 0.012$) and $s$ is the
patient's susceptibility. Each field then updates as:

$$
F(t+1) = F(t) + \text{drive}(t)
$$
$$
D(t+1) = D(t) + \text{drive}(t)
$$
$$
P(t+1) = P(t) + \text{drive}(t)
$$

followed by a monotonicity floor inside the generator itself
($F(t+1) = \max(F(t+1), F(t))$, and likewise for $D$ and $P$) so that floating
point or edge-case arithmetic can never make the generator's own ground truth
non-monotone.

### Biliary strictures (S)

S creeps up slowly every month, but can be relieved by an ERCP procedure. If ERCP
occurs at month $t+1$:

$$
S(t+1) = S(t) - \text{Uniform}(0.08, 0.28)
$$

Otherwise:

$$
S(t+1) = S(t) + \text{S CREEP BASE} + \text{Normal}(0, \text{S CREEP NOISE})
$$

with a monotonicity floor applied only in non-ERCP months.

### Inflammatory activity (A) and cholestasis (C)

Both fields mean-revert toward a shared setpoint $\mu$ ($\text{MEAN AC} = 0.20$)
at a shared speed $k$ ($\text{REVERSION SPEED} = 0.10$), and both receive the
same additive jump when a flare occurs this month:

$$
A(t+1) = A(t) + k(\mu - A(t)) + \text{Normal}(0, \sigma) + \text{flare effect} \cdot \{\text{new flare}\}
$$
$$
C(t+1) = C(t) + k(\mu - C(t)) + \text{Normal}(0, \sigma) + \text{flare effect} \cdot \{\text{new flare}\}
$$

If the patient is a treatment responder and UDCA is active this month, both
fields are additionally suppressed by a multiplicative factor
($\text{UDCA SUPPRESSION} = 0.90$):

$$
A(t+1) \leftarrow A(t+1) \cdot 0.90 \quad \text{(responders only, UDCA active only)}
$$
$$
C(t+1) \leftarrow C(t+1) \cdot 0.90 \quad \text{(responders only, UDCA active only)}
$$

### Flare field

The flare field itself spikes on a new flare event (probability 0.05 per month)
and decays exponentially otherwise:

$$
\text{flare}(t+1) = \text{flare}(t) \cdot \text{decay} + \text{magnitude} \cdot \mathbb{1}\{\text{new flare}\}
$$

with $\text{decay} = 0.65$ and $\text{magnitude} = 0.85$.

### Malignancy hazard (M)

M accumulates as a hazard proportional to the product of F and C, scaled by
susceptibility, using the pre-update values of F and C to keep the causal order
correct:

$$
M(t+1) = M(t) + F(t) \cdot C(t) \cdot \text{rate} \cdot s
$$

with a fixed generator-side rate constant ($\text{M ACCUMULATION RATE} = 0.006$),
clipped to the range $[0, 2]$ and floored at $M(t)$ so it can never decrease.

### Verification

`verify_generator` replays every generated trajectory and checks, transition by
transition, that F, D, P, and M never decrease, that S never decreases outside
ERCP months, and that every field stays within its declared bounds. `train.py`
calls this before any training begins and raises an error if it fails.

## 4. Model architecture (`model.py`, `constraints.py`)

The model is encoder, predictor, decoder, safety net, in a straight pipeline, with
a stop-gradient target encoder used only during training.

```
x(t), context(t) -> Encoder -> z(t) -> Predictor -> z_pred(t+1)
                                                        |
                                                        v
z_pred(t+1), x(t), ercp_flag -> ConstraintAwareDecoder -> x_pred(t+1)
                                                        |
                                                        v
                        x(t), x_pred(t+1), ercp_flag -> SafetyNet -> x_final(t+1)

(training only)
x(t+1), context(t+1) -> Encoder (shared weights, no gradient) -> z_target(t+1)
```

### Encoder

The encoder does not see the raw 8-dimensional state alone. It is given three
hand-specified interaction features, since the generator states these couplings
explicitly and there is no reason to make the network rediscover multiplicative
relationships from scratch:

$$
\text{interactions}(t) = \big[\, F(t) \cdot C(t),\ \text{flare}(t) \cdot A(t),\ \text{flare}(t) \cdot C(t) \,\big]
$$

The encoder input is the concatenation of $x(t)$ (8 dims), $\text{context}(t)$
(6 dims), and $\text{interactions}(t)$ (3 dims), giving a 17-dimensional input,
passed through a two hidden layer MLP (ReLU, hidden width 32) down to a
16-dimensional latent $z(t)$.

### Predictor

The predictor takes $z(t)$ alone (no context is passed in directly; the encoder
has already folded context into $z(t)$) and maps it through a two hidden layer
MLP back to a 16-dimensional $z_{\text{pred}}(t+1)$. This is the core JEPA step:
predicting in latent space rather than in raw state space.

### Constraint-aware decoder

The decoder turns $z_{\text{pred}}(t+1)$ back into an 8-dimensional state,
conditioned on the previous state $x(t)$ and an ERCP flag. See section 5 for
exactly how each field is constrained.

### Safety net

A final deterministic layer that clamps anything that slipped through the
decoder's construction back into a valid range, described in section 5. It also
exposes `count_corrections`, which measures how often it actually had to change
something, as a way of auditing whether the decoder's guarantee is doing its job
unaided.

### Target encoder

During training, the same encoder (shared weights) is run again on the true next
state $x(t+1)$ and $\text{context}(t+1)$, but wrapped in a no-gradient context so
no gradient flows back through this second pass. This produces
$z_{\text{target}}(t+1)$, the target for the predictor's output. This
stop-gradient boundary is the collapse prevention mechanism, described in
section 6.

## 5. Constraint enforcement (`constraints.py`)

Ratchet fields are parameterised as the previous value plus a strictly positive
increment, using the softplus function:

$$
\text{softplus}(h) = \log(1 + e^h) > 0 \quad \text{for all real } h
$$

so that:

$$
F(t+1) = F(t) + \text{softplus}(h_F)
$$
$$
D(t+1) = D(t) + \text{softplus}(h_D)
$$
$$
P(t+1) = P(t) + \text{softplus}(h_P)
$$

This guarantees monotonicity by construction rather than by clipping after the
fact. The distinction matters for gradients: clipping has zero gradient exactly
where the constraint is active (the increment would have been negative), while
softplus is smooth and positive everywhere, so gradient flow is never blocked by
the constraint itself.

M additionally has an upper bound (2.0), enforced with a headroom clamp so the
increment can never push it over the top:

$$
\text{headroom}(t) = \max\big(0,\ M_{\text{upper}} - M(t)\big)
$$
$$
M(t+1) = M(t) + \min\big(\text{softplus}(h_M),\ \text{headroom}(t)\big)
$$

S is handled with a hard branch on the ERCP flag rather than a soft blend, since
ERCP either happens this month or it does not:

$$
S(t+1) = \text{ercp flag} \cdot \sigma(h_{S_{\text{ercp}}}) + (1 - \text{ercp flag}) \cdot \big(S(t) + \text{softplus}(h_{S_{\text{incr}}})\big)
$$

The free fields (A, C, flare) have no monotonicity constraint and are produced
directly by a sigmoid output, bounding them to [0, 1] without any ratchet
structure.

### Safety net (final guarantee)

After the decoder, a deterministic SafetyNet module applies a hard floor against
the previous state on every ratchet field, and a hard bound clamp on every field:

$$
x_{\text{final}}[i] = \max\big(x_{\text{pred}}[i],\ x_{\text{prev}}[i]\big) \quad \text{for } i \in \{F, D, P, M\}
$$
$$
x_{\text{final}}[S] = \max\big(x_{\text{pred}}[S],\ x_{\text{prev}}[S]\big) \quad \text{only in non-ERCP months}
$$
$$
x_{\text{final}} = \text{clamp}\big(x_{\text{final}},\ \text{lower}=0,\ \text{upper}=[1,1,1,1,1,1,2,1]\big)
$$

With the softplus-parameterised decoder, this layer should never actually have to
change anything; `count_corrections` exists to confirm that in practice, and to flag
it immediately if it ever does, which would indicate a decoder bug upstream rather
than the safety net itself failing.

## 6. Representation collapse: prevention and detection

Representation collapse is the failure mode where the encoder maps every input to
nearly the same latent vector. The predictor's loss can still be near zero (it is
easy to predict a constant), but the latent then carries no information about the
actual patient state.

### Prevention mechanism: stop-gradient

The target encoder in the training loop (section 4) is the same network as the
main encoder, run on the true next state, but with gradients blocked. The
predictor is trained to match this fixed target. Because the target encoder's
parameters are not updated through this path, the model cannot cheat by
collapsing both the predictor's output and the encoder's target toward the same
trivial point simultaneously; that symmetric collapse path is broken by
construction.

### Detection mechanism: CollapseMonitor (`anticollapse.py`)

Collapse prevention and collapse detection are deliberately kept separate.
CollapseMonitor is not part of the loss; it is a diagnostic run on a batch of
latent vectors, ideally on held-out data, independent of training.

For a batch of latents $z$ of shape $(n_{\text{samples}}, \text{latent dim})$:

- Per-dimension standard deviation: $\text{std} = z.\text{std}(\dim=0)$. A
  dimension is flagged collapsed if its std falls below a threshold (default
  0.1).
- Effective rank: after centering $z$, take the singular values $\sigma_i$ of
  the centered matrix, normalize them into a probability distribution
  $p_i = \sigma_i / \sum_j \sigma_j$, and compute the exponential of the Shannon
  entropy of that distribution:

  $\text{effective rank} = \exp\Big(-\sum_i p_i \log(p_i)\Big)$

  This is a soft measure of how many independent dimensions the latent is actually
  using. A latent that only truly varies along k directions will have an effective
  rank near k, even if no single dimension's raw standard deviation crosses the
  collapse threshold. This is why both checks are run together: the per-dimension
  std check catches hard collapse (a dimension going flat), and effective rank
  catches soft collapse or redundancy (many dimensions moving together without
  adding independent information).

The monitor produces a verdict of SEVERE COLLAPSE, PARTIAL COLLAPSE, REDUNDANCY
WARNING, or HEALTHY based on these two measurements together, and is run both
during training (on validation latents, every epoch, printed periodically) and
during evaluation (on a sample of held-out latents).

## 7. Loss functions (`train.py`)

The total training loss has three components.

### Latent loss (core JEPA objective)

$$
\mathcal{L}_{\text{latent}} = \text{MSE}\big(z_{\text{pred}}(t+1),\ z_{\text{target}}(t+1)\big)
$$

where $z_{\text{target}}(t+1)$ comes from the stop-gradient target encoder.

### Reconstruction loss

$$
\mathcal{L}_{\text{recon}} = \text{L1}\big(x_{\text{final}}(t+1),\ x_{\text{true}}(t+1)\big)
$$

an L1 loss on the fully decoded and safety-netted state against the ground truth
next state.

### Coupling loss (auxiliary)

The generator states three couplings explicitly: M accumulates as a hazard of
sustained F times C, flares perturb A and C together, and treatment suppresses A
and C for responders. The first of these is reinforced with an explicit auxiliary
loss term, since it is the one coupling that is a longer-horizon accumulation
rather than an instantaneous effect and is not otherwise directly supervised at
every step in the same way:

$$
\Delta M_{\text{pred}} = x_{\text{final}}[M](t+1) - x_{\text{prev}}[M](t)
$$
$$
\Delta M_{\text{target}} = x_{\text{prev}}[F](t) \cdot x_{\text{prev}}[C](t) \cdot e^{\,\text{log m rate}}
$$
$$
\mathcal{L}_{\text{coupling}} = \text{MSE}\big(\Delta M_{\text{pred}},\ \Delta M_{\text{target}}\big)
$$

`log_m_rate` is a single learned scalar parameter owned by the model, stored
in log space so its exponent is always positive, initialized to a small
uninformed guess. It is never given the generator's true rate constant directly;
the model has to learn its own estimate purely from the gradient of this loss
term. This is deliberate: handing the model the true rate would be leaking the
answer rather than requiring it to be learned.

### Combined loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{latent}} + \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}} + \lambda_{\text{coupling}} \cdot \mathcal{L}_{\text{coupling}}
$$

with $\lambda_{\text{recon}} = 1.0$ and $\lambda_{\text{coupling}} = 0.3$
(coupling is a weak auxiliary signal,
not a primary objective, partly because the true coupling target itself is only
approximate: the generator's real per-step M increment also depends on the hidden
per-patient susceptibility, which the loss's target formula does not have access
to).

## 8. Training procedure (`train.py`)

### Data

500 training patients and 100 validation patients are generated fresh each run
(24-month trajectories), with disjoint seed ranges so there is no overlap between
splits.

### Multi-step windows with scheduled sampling

Rather than only training on single $(x(t), x(t+1))$ pairs, training unrolls a
short window (default 3 steps) starting from a random point in each trajectory.
At each step within the window, the model either receives the true previous
state as its next input (teacher forcing) or feeds its own predicted state back
in (scheduled sampling), controlled by a decaying teacher-forcing probability:

$$
\text{tf prob}(\text{epoch}) = \max\big(\text{floor},\ \text{start} \cdot \text{decay}^{\text{epoch}}\big)
$$

with $\text{start} = 1.0$, $\text{decay} = 0.97$, $\text{floor} = 0.3$. Early in
training the model almost
always sees ground truth as input, which is an easier optimization landscape;
later in training it increasingly has to cope with its own prediction errors
compounding across the window, which is closer to how it will actually be used at
inference time (autoregressive rollout).

### Optimization

Adam optimizer, learning rate 1e-3, gradient norm clipped to 1.0, with a
ReduceLROnPlateau scheduler (patience 6, factor 0.5) driven by validation MAE. The
best validation checkpoint (lowest mean absolute error on the validation set) is
saved to disk; training history (validation MAE, teacher-forcing probability, and
latent effective rank per epoch) is saved to a JSON file for later plotting.

### Validation and collapse monitoring during training

After every epoch, the model is run once (no teacher forcing, single-step) over
the entire validation set to compute validation MAE, and a sample of validation
latents is passed through CollapseMonitor to track effective rank across
training, which is what gets plotted alongside validation MAE in the training
curve figure.

## 9. Evaluation methodology (`evaluate.py`)

Four distinct evaluation angles, each answering a different question.

### Predictive accuracy

Rolling forecast evaluation: give the model the first N months of history
(default 12), then let it roll forward autoregressively (`model.rollout`) for the
remainder of a held-out trajectory, using its own predictions as input at every
step (no teacher forcing at all). Mean absolute error is reported overall and
broken down per field.

### Constraint violation rate

For every transition in a held-out set, checks whether the raw decoder output
(`x_pred`, before the safety net) ever violates a ratchet field's monotonicity,
and separately whether the fully safety-netted output (`x_final`) ever does. The
two
numbers answer different questions: a violation in the pre-safety-net output that
disappears post-safety-net means the guarantee is coming entirely from the final
clamp rather than from the decoder's construction; a violation that survives the
safety net would indicate the safety net itself has a bug.

### Coupling check

Separate from monotonicity. Compares the model's predicted M increment against
the generator's true rate formula
($F(t) \cdot C(t) \cdot \text{true rate} \cdot \text{susceptibility}$), which is
only available at evaluation time since it depends on the hidden susceptibility
value the generator does not expose to the model. Also reports the model's
learned `log_m_rate` against the generator's true constant for direct
comparison.
This tests whether the accumulation dynamics are faithful, independent of whether
the monotonicity shape constraint holds.

### Generalisation probes

Three probes, each changing one axis relative to the training distribution while
holding the generator's underlying rules fixed:

- Probe 1, high susceptibility: patients drawn with susceptibility restricted to
  [0.80, 1.00] instead of the training range [0.30, 1.00], testing whether the
  model can extrapolate to faster-progressing patients purely from what a single
  timestep implies about their trajectory shape, since the model never observes
  susceptibility directly.
- Probe 2, late treatment start: UDCA start month drawn from [13, 19) instead of
  the training range [2, 9), testing whether the model has learned a general
  "suppress A and C once UDCA is active" rule versus a timing-specific prior.
- Probe 3, longer rollout: 48-month trajectories evaluated with a model trained
  only on 24-month windows, testing long-horizon autoregressive stability, where
  compounding error and any systematic rate bias (such as in the M coupling)
  would be expected to show up most clearly.

All three probes remain within the same generator; this establishes generalisation
within the synthetic process only, not validity against a real disease.

### Explained prediction

`explain_prediction` traces one specific rollout prediction back to the summary
statistics of the patient's own history (mean F, mean C, mean F times C over the
history window) and the patient's context (responder status, UDCA start month),
producing a note about why the model's predicted trajectory looks the way it
does. This is the mechanism behind the memo's "why did the model predict this?"
example.

## 10. Repository structure

```
.
├── generator.py            synthetic data generator and its own constraint check
├── model.py                encoder, predictor, LiverWorldModel, context encoding
├── constraints.py          ConstraintAwareDecoder and SafetyNet
├── anticollapse.py         CollapseMonitor diagnostic
├── train.py                training loop, loss functions, scheduled sampling
├── evaluate.py              evaluation harness (accuracy, violations, coupling,
│                            generalisation probes, explained prediction)
├── plots.py                 generates the three figures used in the memo
├── visualize_generator.py   sanity-check plots for the generator alone
└── memo.pdf                 decision memo written for a staff-engineer audience
```

## 11. How to run

Dependencies: PyTorch, numpy, and matplotlib. No other external packages are
required.

1. Generator sanity check (recommended before training):

   ```
   python visualize_generator.py
   ```

   Produces five plots checking that ratchet fields are monotone, that responders
   show suppressed A and C after treatment starts, that ERCP produces the
   expected step-down in S, that flares visibly perturb A and C, and that the M
   increment tracks the F times C product.

2. Train:

   ```
   python train.py
   ```

   Runs `verify_generator` first and raises an error if the generator violates
   its own constraints. Trains for 80 epochs by default, printing progress every
   10 epochs. Saves the best checkpoint to `best_model.pt` and per-epoch history
   to `training_history.json`.

3. Evaluate:

   ```
   python evaluate.py
   ```

   Loads `best_model.pt` and runs predictive accuracy, constraint violation rate,
   the coupling check, a collapse report on held-out latents, all three
   generalisation probes, and one explained prediction, printing everything to
   stdout.

4. Plots:

   ```
   python plots.py
   ```

   Requires `training_history.json` and `best_model.pt` to already exist (run
   `train.py` first). Produces the training curve, a rollout example figure, and
   the generalisation probe bar chart.

## 12. Design rationale

### Why a JEPA-style latent, and what it was weighed against

Two simpler alternatives were considered and set aside:

- Using x(t) directly as the latent makes the constraint story trivial (the
  "latent" is already the constrained state), but gives the model nowhere to
  represent anything not directly observable, most importantly the hidden
  per-patient susceptibility that scales both the ratchet drive and the M rate.
  That signal has to live somewhere other than the raw 8 numbers if it is going to
  be used at all.
- A plain Neural-ODE suits continuous, smooth drift well, but flares are discrete
  monthly events that jump the flare field and perturb A and C together, and ERCP
  is a discrete, non-differentiable branch in how S behaves. Forcing that into a
  smooth ODE means either smoothing away the events, losing them, or bolting on a
  separate discrete-event handler on top, losing most of the simplicity benefit.

A learned predictive latent is more expressive than either alternative, since it
can carry information like inferred susceptibility that the raw state does not
contain, but it is also freer to drift off the manifold of valid states and
carries a real risk of collapse. That is why sections 5 and 6 above exist as
first-class parts of the design rather than afterthoughts.

### Why explicit interaction features and an auxiliary coupling loss, rather than a learned attention mechanism over a causal graph

The generator states exactly three couplings explicitly: F times C drives M,
flares perturb A and C together, and treatment suppresses A and C for responders.
Hand-specifying these three interaction terms as encoder inputs, plus one
auxiliary loss term for the accumulation coupling, matches what is actually
specified without inventing additional structure the generator does not have. A
graph-attention encoder would need a real causal graph with meaningfully more than
three edges to justify its added complexity and reduced interpretability relative
to a plain MLP with explicit features.

### Why scheduled sampling instead of full backpropagation-through-time

Pure teacher forcing never exposes the model to its own mistakes during training,
which is a mismatch with how the model is actually used at inference time
(autoregressive rollout). Scheduled sampling, with a decaying probability of using
ground truth versus the model's own prediction as the next input, is a practical
middle ground within the project's time budget. Full backpropagation through the
entire unrolled sequence is more principled and is the natural next step to try.