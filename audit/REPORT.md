# FAIRFL paper audit: claims vs. code and runs

Paper: *FAIRFL: Blockchain-based Enforcement of Valid Contributions in Federated Learning* (`polfl.pdf`).
Code audited: `main` at `d702bb6` (checked out as a separate worktree at `../IHF-DFL-main-audit`).
Runs audited: all 480 runs in `data/runs/` (read-only).
Audit date: 2026-10-27.

## Scope and method

- The current branch `BugfixOctober` points to the same commit as `main` (`d702bb6`), so `git diff main...HEAD` is empty. No finding below has been changed on the current branch.
- The run outputs record no commit hash. No commit on any branch touches `src/`, `contracts/` or `experiment/` between the first run (2026-05-22 12:05) and the last run (2026-06-01). The last code commit on `main` (`83b8bb5`, 2026-05-22 12:01) landed four minutes before the first run. The runs were therefore very likely produced from code equal to `main`, but this cannot be proven (see F7).
- Every reported number was recomputed from the `.pkl` files with scripts in `audit/`. No source file was modified and no experiment was rerun.

### Reproducing the audit

Run from the repository root (needs pandas, numpy, torch and web3; the tests import from the `main` worktree):

```
python audit/load_all.py          # loads all 480 pickles, caches to audit/cache.pkl
python audit/recompute.py         # headline numbers           -> audit/recompute.out
python audit/check_scoring.py     # normalization used         -> audit/check_scoring.out
python audit/misc_checks.py       # filtering, round 1, stats  -> audit/misc_checks.out
python -m pytest audit/test_audit_repro.py -v -p no:cacheprovider --rootdir=audit
```

The removal-cause table (F3) was produced by an inline script whose output is saved in `audit/removal_cause.out`.

### Definitions used when recomputing

- **Permanent removal:** a participant whose `state` becomes `disqualified` in any round. The removal round is the first such round.
- **Post-activation delay (Fig. 3):** removal round minus free-rider activation round. This definition reproduces every median in Fig. 3.
- **Final accuracy:** `objective_global_accuracy` at the last round (10 for MNIST, 25 for CIFAR-10), unweighted mean over runs.
- **Round-one gas:** sum of `gas_used` over all receipts with `round == 1`, averaged over runs.
- **Within-round score variance:** population variance of the stored contribution scores in a round, averaged over the rounds of a run.
- **Counts:** pooled over the ten repetitions and, where the paper pools, over activation rounds.

## Summary

Every number printed in the paper reproduces exactly from the stored runs. The problems are in what those numbers mean:

1. **The malicious participant never acted maliciously in any of the 480 runs** (F1). It trained, voted and reported honestly in all 8,880 participant-rounds. Every result about malicious actors (RQ1a, Fig. 4, the "Malicious" curves in Fig. 5) is data about an honest participant. As a result, the honest-removal rates are under-reported.
2. **The Loss and Accuracy strategies do not use the paper's Eq. 7** (F2). They use a different, clamped normalization. The paper's explanation for why honest participants were removed (sign inversion under signed normalization) cannot apply to them.
3. **Settlement does not follow Eq. 11** (F3). A contribution score below −3 removes a participant in a single round, whatever their reputation. This undocumented rule causes most of the Accuracy+Loss honest removals.

## Part 1: Claims ledger

Verdicts: Supported / Partially supported / Unsupported / Contradicted / Not checkable.

### Setup claims

| ID | Claim | Location | Verdict | Evidence |
|---|---|---|---|---|
| S1 | MNIST and CIFAR-10, IID partitions among participants | §3.4 | Supported | `src/ml/data.py:216-230`: equal-size `random_split` seeded by run_id via `data/seeds.py`. |
| S2 | Static parameters: epochs 1/25, batch 32/128, 4/6 honest, 1 malicious, 1 free-rider, reward 1 ETH, collateral 1 ETH, rounds 10/25, punish divisor 3, first-round fee 50% | Table 3 | Partially supported | Run metadata matches every value in all 480 runs (`recompute.out`, "Config sanity"). However, the "malicious" participant behaved honestly, so the effective number of honest participants is 5 (MNIST) and 7 (CIFAR-10). See F1. |
| S3 | 4 strategies × 2 noise levels × 3 activation rounds, 10 repetitions, 480 runs | Table 4, §3.4 | Supported | 480 `.pkl` files, exactly 10 per cell. Filename and metadata agree in 480/480 runs. |
| S4 | Report filtering enabled for the metric strategies | Table 4 | Supported | `use_outlier_detection=True` in all runs; filtering is always applied in the metric strategies. The threshold τ is not reported (F5). |
| S5 | The malicious actor votes −1 for all honest participants and submits noise-injected weights from the first round | §3.4 | **Contradicted** | Role `bad` has behavior `good` in 8880/8880 participant-rounds; `attack_type` is empty for every row. See F1. |
| S6 | The free-rider trains honestly until its activation round, then submits the previous global model plus uniform noise | §3.4 | Partially supported | Behavior switches exactly at the configured activation round in 480/480 runs; noise is uniform in ±scale (`src/ml/attacks.py:11-21, 66-85`). In round 1, the free-rider's model is its own randomly initialized network, not the previous global model (F4). |
| S7 | Participants not merged in round 1 receive an increased punishment (first-round fee) | §3.4 | Supported | `contracts/OpenFLModel.sol`: `collectFreeriderFees`, `paybackFreeriders`. |
| S8 | Executed on a local Ethereum fork | §3.4 | Supported | `fork=True` in all runs. |
| S9 | MNIST and low-noise CIFAR-10 on a Tesla T4; high-noise CIFAR-10 on an AMD Radeon RX 7900 XTX | §3.4 | Supported | `gpu_name` in each run's setup. |
| S10 | Each configuration repeated ten times | §3.4 | Partially supported | Ten runs exist. Only the data split is seeded; model initialization and training are unseeded. The 24 configurations that share a run_id show 18 to 24 distinct round-0 global accuracies (`misc_checks.out`). |

### Method claims

| ID | Claim | Location | Verdict | Evidence |
|---|---|---|---|---|
| M1 | Vote 1 if the model is up to 7% worse "than before", 0 up to 14%, −1 beyond | §2.3 | Partially supported | `src/ml/evaluation.py:113-128`: thresholds are absolute percentage points, compared with the evaluator's own local validation accuracy, not with a previous model. |
| M2 | RRS as a GRS-weighted sum of votes (Eq. 1); punishment GRS/punish_factor (Eq. 3) | §2.3 | Supported | `OpenFLModel.sol` `feedback` (lines 297-330) and `punishMaliciousUsers` (lines 413-445). |
| M3 | Merged set M = participants with RRS ≥ 0, aggregated by FedAvg weighted by local dataset size | §3.2 | Partially supported | `src/ml/aggregation.py:224-225` uses an unweighted mean. This is equivalent here because the partitions are equal in size. When every RRS is negative, all participants are merged (`src/contracts/fl_challenge.py:800-802`); the paper does not mention this. |
| M4 | Evaluators submit a_{j,i}, ℓ_{j,i}, A_prev and L_prev with the feedback array | §3.2.2, Table 2 | Supported | `src/contracts/fl_challenge.py:321-358`. |
| M5 | Robust z-score filter (Eq. 4) with threshold τ and ε guarding a zero MAD; mean of the filtered reports (Eq. 5) | §3.2.2 | Partially supported | `src/contracts/contribution.py:453-496`. τ = 1.1 is not reported in the paper. When MAD = 0, the code keeps reports within 1.1 raw units of the median instead of using ε (lines 469-479). See F5. |
| M6 | "The evaluated prototype normalizes improvements by their signed sum" (Eq. 7) | §3.2.2 | **Contradicted** for Loss and Accuracy; supported for Accuracy+Loss | Stored Loss and Accuracy scores match `normalize_contribution_scores_new` in 100% of 4,194 checked rounds and Eq. 7 in only about 38%. See F2. |
| M7 | c_loss = n(d_L), c_acc = n(d_A), c_acc+loss = ½(c_acc + c_loss) (Eq. 8-10) | §3.2.3 | Partially supported | Eq. 10 with Eq. 7 reproduces the stored Accuracy+Loss scores in 2049/2049 rounds. Eq. 8 and Eq. 9 do not hold (F2). |
| M8 | Naive: every merged participant gets 1/N | §3.2.1 | Supported | All scores in a round are equal in 2100/2100 Naive rounds. |
| M9 | Settlement Eq. 11: GRS_i += R·w_i / Σ max(w_k, 0); participants with a negative score lose GRS through this formula | §3.3 | **Contradicted** | `OpenFLModel.sol:529-552`: a negative score costs GRS·\|c\|/punish_factor_contrib, and the participant is disqualified if \|c\| > 3. See F3. |
| M10 | A participant is excluded when its GRS falls below min_collateral/punish_factor | §3.3 | Partially supported | Also excluded immediately when \|c\| > punish_factor_contrib = 3 (`OpenFLModel.sol:535`). See F3. |
| M11 | Scoring: a single participant computes and submits the scores; any other can verify | §3.3 | Partially supported | `src/contracts/contribution.py:49-70`: every merged participant submits its own score in its own transaction. |
| M12 | Reports are encoded as fixed-point integers scaled by 10^18 | §3.3 | Partially supported | Accuracy reports are scaled by 10^4 and losses by 10^2 with a uint16 cap of 65535 (`src/ml/evaluation.py:60-61, 170-179`). Only contribution scores use 10^18 (`contribution.py:51`). |
| M13 | Dispute settlement is not implemented; submitted scores are assumed correct | §3.3 | Supported | No dispute logic exists in the contract or the Python code. |

### Result claims

| ID | Claim | Location | Verdict | Recomputed vs reported |
|---|---|---|---|---|
| R1 | Low-noise free-riders permanently removed: Loss 59/60, Accuracy 51/60, Accuracy+Loss 27/60, Naive 8/60 | Abstract, §4.1, §5, §6 | Supported | 59 / 51 / 27 / 8 |
| R2 | At noise 0.1, every strategy removed every free-rider | §4.1, §5 | Supported | 60/60 for each strategy |
| R3 | CIFAR-10 low noise: Loss and Accuracy removed every free-rider; Naive removed none | §4.1 | Supported | 30/30, 30/30, 0/30 |
| R4 | Accuracy+Loss at activation rounds 3 and 5: 1/10 and 1/10 on CIFAR-10; 4/10 and 1/10 on MNIST | §4.1 | Supported | Exact match |
| R5 | Loss median delay reaches 6.5 rounds (CIFAR-10, activation round 3) | §4.1, §5 | Supported | 6.5 under the delay definition above. The free-rider is exposed for one more round than the delay states (F8). |
| R6 | Fig. 3: all 48 cells (removal counts and median delays) | Fig. 3 | Supported | Every count and median matches (`recompute.out`). |
| R7 | Malicious actors removed: MNIST Accuracy+Loss 4/60, others 0; CIFAR-10 Accuracy 11/60, Accuracy+Loss 10/60, Loss 7/60, Naive 0 | §4.2, Fig. 4 | **Contradicted** as evidence about malicious actors | The counts reproduce exactly, but the participant never behaved maliciously (F1). |
| R8 | Honest participants removed: MNIST Accuracy+Loss 19/240, Accuracy 1/240, Loss 1/240, Naive 0; CIFAR-10 55/360, 43/360, 50/360, 0 | §4.2 | Partially supported | The counts reproduce, but they are under-reported because the "malicious" participant was also honest. Corrected: MNIST 23/300, 1/300, 1/300, 0/300; CIFAR-10 65/420, 54/420, 57/420, 0/420. |
| R9 | Fig. 5: Loss drives the low-noise free-rider's GRS toward zero on both datasets; Naive preserves or increases it | §4.3, §5 | Partially supported | Final mean free-rider GRS under Loss is 0.00 ETH on both datasets. Under Naive on MNIST it ends at 0.76 ETH, below the 1.0 ETH start and below the honest mean of 1.25 ETH. The "Malicious" curves show an honest participant (F1). |
| R10 | Every stored contribution-score vector sums to one | §4.3 | Supported | Maximum \|Σ − 1\| = 2.3 × 10^-13 |
| R11 | Accuracy+Loss scores range from −19.06 to 24.15 (MNIST) and −40.54 to 48.60 (CIFAR-10) | §4.3 | Supported | Exact match |
| R12 | Fig. 6: zero variance for Naive, moderate for single metrics, long-tailed for Accuracy+Loss | §4.3, Fig. 6 | Supported | Naive 0. Single-metric medians 0.26 to 0.59. Accuracy+Loss maxima 13.2 (MNIST) and 54.3 (CIFAR-10). |
| R13 | Such magnitudes arise when signed improvements nearly cancel in the denominator of Eq. 7 | §4.3 | Partially supported | True for Accuracy+Loss. Loss and Accuracy scores are clamped at −1 and do not use Eq. 7 (F2). |
| R14 | Final accuracy: MNIST about 94.4% for every strategy; CIFAR-10 64.6% Naive, 65.3% Accuracy, 65.0% Loss, 64.6% Accuracy+Loss | §4.4 | Supported | MNIST 94.39 to 94.45%. CIFAR-10 64.6 / 65.3 / 65.0 / 64.6%. |
| R15 | Round-one gas vs Naive: MNIST +39.8% Accuracy, +40.1% Loss, +52.9% Accuracy+Loss; CIFAR-10 +38.6%, +39.6%, +49.1% | Abstract, §4.5 | Supported | Exact match |
| R16 | The additional metric reports dominate the gas overhead | §4.5 | Supported | Round-one feedback gas on MNIST rises from 1.85M (Naive) to 3.23M to 3.68M; the other transaction types barely change. |
| R17 | Loss is the most informative signal and outperforms Accuracy | §5, §6 | Partially supported | The gap comes entirely from MNIST: 29/30 vs 21/30 (Fisher two-sided p = 0.012). On CIFAR-10 both remove 30/30 (F6). |
| R18 | Accuracy removed 21/30 MNIST low-noise free-riders | §5 | Supported | 21/30 |
| R19 | Malicious updates were merged in most rounds because they did not degrade the model enough to fail the vote | §5 (RQ1a) | **Contradicted** | No malicious update was ever submitted (F1). |
| R20 | Honest removals are attributed mainly to signed normalization: n(d) = n(−d) can flip the sign of every score | §5 (RQ2) | **Contradicted** for Loss and Accuracy; partially supported for Accuracy+Loss | Loss and Accuracy do not use Eq. 7 (F2). For Accuracy+Loss, the removals happen through the undocumented \|c\| > 3 rule (F3). |
| R21 | Accuracy+Loss performs worse because averaging dilutes the loss signal | §5 | Partially supported | Eq. 10 matches the code. Its high honest-removal count comes from the \|c\| > 3 rule (F3). |
| R22 | Contribution scoring did not change final accuracy | §4.4, §5 | Supported | Means within 0.7 percentage points per dataset. |
| R23 | FAIRFL changes rewards but not which updates are aggregated, which is decided by the OpenFL vote | §5 | Supported | The merge set depends only on RRS (`fl_challenge.py:800`). |
| R24 | About 10^4 reports per metric and round for 100 participants | §5 | Not checkable | Analytic estimate, not measured. |
| R25 | Malicious participants are rarely removed | Abstract, §6 | **Contradicted** | Unsupported by the runs; see F1. |

## Part 2: Findings

Ordered by impact on the paper's conclusions. "Changed on current branch" is no for every finding, because `BugfixOctober` points to the same commit as `main`.

### F1. The malicious participant is never activated

- **Type:** both (code bug and paper hole)
- **Status:** Confirmed
- **Severity:** Critical
- **Affected claims:** S2, S5, R7, R8, R9, R19, R25; Fig. 4 and Fig. 5
- **Location:**
  - `experiment/experiment_presets.py`: the four paper presets (`mnist_openfl_low_noise`, `mnist_openfl_high_noise`, `cifar_openfl_low_noise`, `cifar_openfl_high_noise`) set `malicious_start_round=None` and `malicious_noise_scale=None`.
  - `experiment/experiments.py:95-104` turns these into `[None]` and iterates them; lines 217-218 assign the per-run `None` to the configuration. This overrides the fallback in `ExperimentConfiguration.__init__`, which had already set the values to the free-rider's. The helper `resolve_attack_params` (`experiment/helper.py`) would also resolve them, but it is only used for skip detection.
  - `src/ml/pytorch_model.py:125` passes `_attitudeSwitch = self.malicious_start_round`, which is `None`.
  - `src/ml/attacks.py:90` switches attitude only when `user.attitudeSwitch == pm.round`. `None == round` is never true, so the participant stays `good` forever.
- **Evidence:**
  - `audit/test_audit_repro.py`. Six tests fail for the predicted reason. `test_malicious_participant_switches_to_bad_by_round_1` runs `update_users_attitude` for rounds 1 to 25 and fails with `assert 'good' == 'bad'`. The four preset tests fail with `assert None is not None`.
  - `audit/recompute.out`, "Did the malicious participant ever behave maliciously?": role `bad` has behavior `good` in 8880/8880 participant-rounds, and `attack_type` is empty in every row.
  - Every run CSV logs the participant at round 0 as `$user$4, 0, good, bad, None, ...` (attitude, future attitude, switch round).
- **Explanation:** Because its attitude stays `good`, the "malicious" participant trains normally, votes using the honest thresholds, and reports true metrics. The voting logic (`evaluation.py:76`) and the noise injection (`attacks.py:46`) both key on the current attitude, so neither attack ever runs. Consequences:
  - Every malicious-actor statistic in the paper describes an honest participant. Its removal rates (for example 11/60 under Accuracy on CIFAR-10, 18%) are close to the honest rates (43/360, 12%), as expected.
  - The RQ1a discussion ("the malicious updates were merged in most rounds, i.e., they did not degrade the model enough to fail the vote") explains an attack that did not happen.
  - The experiment tested no poisoning at all, not "weak" poisoning.
  - Honest-removal rates are under-reported: the corrected values are given in R8.
  - If the attitude had switched, `manipulate()` would have been called with `scale=None` and raised `TypeError`, so the code path for these presets could not have worked either way.
- **Suggested remedy:** Resolve the malicious parameters before building `PytorchModel` (call `resolve_attack_params` in `experiments.py`, or skip the per-run override when the preset leaves them unset). Make `PytorchModel` raise an error when a role is present but its activation round is `None`. Then rerun with the paper's stated behavior (start round 1) and a malicious noise scale chosen and reported in the paper. Alternatively, drop RQ1a and relabel the participant as honest (see Part 4).
- **Changed on current branch:** no

### F2. The Loss and Accuracy strategies do not use Eq. 7

- **Type:** both
- **Status:** Confirmed
- **Severity:** High
- **Affected claims:** M6, M7, R13, R20
- **Location:** `src/contracts/contribution.py:254` (Accuracy) and `:313` (Loss) call `normalize_contribution_scores_new` (defined at `:409`). That function:
  1. handles rounds where every improvement is negative by replacing each value with sum/value, which makes every score positive;
  2. rescales negative values so that the minimum is exactly −1;
  3. adds the excess to the positive values in proportion so that the scores sum to one.
  Only Accuracy+Loss uses the signed-sum normalization of Eq. 7 (`normalize_contribution_scores_old`, `:187-190`).
- **Evidence:**
  - `audit/check_scoring.out`: re-applying each normalization to the stored filtered averages (`user_mad_avg`, `prev_avg_round_val_after_mad`) reproduces the stored scores with `_new` in 100% of 4,194 Loss and Accuracy rounds, and with Eq. 7 in only 38.2% (Accuracy) and 38.8% (Loss). The Eq. 7 matches are rounds where every improvement is positive, where both methods agree.
  - In 758 Accuracy rounds and 785 Loss rounds the improvements sum to a negative number; Eq. 7 would invert every sign in those rounds, whereas `_new` does not.
  - `audit/test_audit_repro.py::test_single_metric_normalization_is_signed_sum_eq7`: for improvements [1, 2, −10, −5] the code returns `[0.83, 1.67, -1.0, -0.5]`; Eq. 7 gives `[-0.08, -0.17, 0.83, 0.42]`.
- **Explanation:** §3.2.2 states that the evaluated prototype normalizes by the signed sum, and §4.3 and §5 build on that: unbounded scores, sign inversion, and n(d) = n(−d) as the main cause of honest removals. For Loss and Accuracy, scores are bounded below by −1 and signs are preserved. The explanation offered for the 43 to 55 honest CIFAR-10 removals therefore does not apply to these two strategies. Their honest removals come from repeated punishments of GRS·\|c\|/3 until GRS reaches 1/3 ETH (see F3). The numbers do not change; the method description and the causal explanation do.
- **Suggested remedy:** Either describe `normalize_contribution_scores_new` in §3.2.2 and rewrite Eq. 8, Eq. 9 and the RQ2 discussion, or switch the Loss and Accuracy strategies to Eq. 7 and rerun. Using one normalization for all three metric strategies would also make the comparison between them fair.
- **Changed on current branch:** no

### F3. Settlement does not follow Eq. 11; a score below −3 removes a participant at once

- **Type:** paper hole
- **Status:** Confirmed (from the stored on-chain outcomes and the traced contract path; the Solidity test suite was not run because `forge` tests are skipped on Windows)
- **Severity:** High
- **Affected claims:** M9, M10, R8, R20, R21
- **Location:** `contracts/OpenFLModel.sol:529-552`, `settleContributionScores`:
  - `punishment = (GRS / punishfactorContrib) * |c| / 1e18`, which is GRS·\|c\|/3;
  - the participant is disqualified if `GRS <= min_collateral / punishfactorContrib` **or** `punishment > GRS`. The second condition is equivalent to \|c\| > 3.
  - Positive scores are rewarded as `reward · w_i / Σ w_k` over positive weights only (lines 515-527, 557-560), as in Eq. 11.
- **Evidence:** `audit/removal_cause.out` classifies every permanent removal by its cause:

  | Dataset | Strategy | Role | Not merged (RRS < 0) | Score < −3 (instant) | Score < 0 and GRS ≤ 1/3 ETH |
  |---|---|---|---|---|---|
  | CIFAR-10 | Accuracy+Loss | honest | 0 | 45 | 10 |
  | CIFAR-10 | Accuracy+Loss | "malicious" | 0 | 9 | 1 |
  | CIFAR-10 | Accuracy | honest | 0 | 0 | 43 |
  | CIFAR-10 | Loss | honest | 0 | 0 | 50 |
  | MNIST | Accuracy+Loss | honest | 0 | 14 | 5 |
  | MNIST | Accuracy+Loss | "malicious" | 0 | 4 | 0 |
  | MNIST | Accuracy | honest | 0 | 0 | 1 |
  | MNIST | Loss | honest | 0 | 0 | 1 |

  In total, 72 of the 88 Accuracy+Loss removals of non-free-riders happen in a single round with c < −3. Median GRS just before such a removal was 1.22 ETH (minimum 0.29, maximum 3.15), so participants with above-average reputation were removed by one bad round.
- **Explanation:** Eq. 11 says a negative score reduces GRS through R·w_i/Σ max(w_k, 0), and §3.3 names only the GRS < min_collateral/punish_factor threshold as an exclusion rule. Neither the GRS·\|c\|/3 penalty nor instant removal at \|c\| > 3 is described. The \|c\| > 3 rule turns Accuracy+Loss's unbounded scores (F2, R11) directly into removals. This drives the "Accuracy+Loss is less reliable" conclusion and contradicts the use-case goal that honest participants are protected by accumulated reputation (§3.1).
- **Suggested remedy:** Document the actual penalty and both exclusion conditions in §3.3 and attribute the Accuracy+Loss honest removals to them. Alternatively, align the contract with Eq. 11 (or bound the per-round penalty) and rerun.
- **Changed on current branch:** no

### F4. Round 1 averages independently initialized models

- **Type:** both
- **Status:** Confirmed
- **Severity:** Medium
- **Affected claims:** S6, M3; round-1 values in Fig. 5 and Fig. 7
- **Location:** `src/ml/pytorch_model.py:112-116` creates a fresh `Net_MNIST()` or `Net_CIFAR()` for every participant. The global model is created separately at `:35-37`. No weights are copied before round 1, and torch is not seeded.
- **Evidence:** `audit/misc_checks.out`:

  | Dataset | Mean honest local validation accuracy, round 1 | Global accuracy, round 0 | Global accuracy after round-1 merge | Global accuracy after round 2 |
  |---|---|---|---|---|
  | MNIST | 68.9% | 10.2% | 25.7% | 63.9% |
  | CIFAR-10 | 28.6% | 10.0% | 10.0% | 20.0% |

  Averaging unrelated networks destroys most of what each participant learned. On CIFAR-10 the merged model stays at chance after 25 local epochs.
- **Explanation:** Background §2.1 describes FL as sending one initial model to every user, but the prototype does not. Consequences: round 1 is mostly wasted (worst on CIFAR-10); round-1 improvements are measured against a random global model; and a free-rider that activates in round 1 submits its own random network plus noise rather than "the previous global model". The effect applies to all strategies equally, so the comparative conclusions are likely unaffected.
- **Suggested remedy:** Load the global model's state into every participant before round 1, and seed torch per run. Otherwise, state the behavior in §3.4.
- **Changed on current branch:** no

### F5. The filter threshold τ is unreported and discards about a quarter of all reports

- **Type:** paper hole
- **Status:** Confirmed
- **Severity:** Medium
- **Affected claims:** S4, M5
- **Location:** `src/contracts/contribution.py:136, 216, 276` set `mad_threshold=1.1`. The robust z-score of Leys et al. (2013), cited in the paper, is normally used with a cutoff around 2.5 to 3.5. The MAD = 0 branch (`:469-479`) keeps reports within 1.1 raw units (0.011 percentage points of accuracy, or 0.011 loss) of the median instead of applying ε.
- **Evidence:** `audit/misc_checks.out`: 26.6% to 28.5% of peer reports are discarded per participant per round, in every dataset, strategy and metric. In these runs no evaluator reported dishonestly (F1; the free-rider reports true metrics of the previous global model), so the filter is mostly discarding honest reports.
- **Explanation:** τ shapes every metric-based score, yet it is neither reported nor justified. With five to seven reports per participant, discarding about 27% means the average often rests on three or four values.
- **Suggested remedy:** Report τ = 1.1 and the MAD = 0 rule in §3.2.2, justify the value, and preferably show sensitivity to τ.
- **Changed on current branch:** no

### F6. "Loss is more informative than Accuracy" rests on MNIST alone

- **Type:** paper hole
- **Status:** Confirmed
- **Severity:** Low
- **Affected claims:** R17
- **Evidence:** `audit/misc_checks.out`: on MNIST, Loss removes 29/30 low-noise free-riders and Accuracy 21/30 (Fisher two-sided p = 0.012). On CIFAR-10 both remove 30/30 (p = 1.0). Pooled: 59/60 vs 51/60 (p = 0.017).
- **Explanation:** The comparison is genuine, but it comes from one dataset, ten repetitions per cell, and the late-activation conditions. Partly faster removal on CIFAR-10 is not tested.
- **Suggested remedy:** Report the per-dataset counts with the test, and qualify the claim to "on MNIST" or "in the evaluated setting".
- **Changed on current branch:** no

### F7. The runs cannot be reproduced exactly from `main`

- **Type:** paper hole
- **Status:** Confirmed
- **Severity:** Low
- **Affected claims:** S10; all results indirectly
- **Location:** `experiment/experiments.py:38-40` hard-codes `preset = "test"` and `datasets = [ DATASETFAST ]`, so producing the paper runs required uncommitted local edits. No commit hash or working-tree state is written to the `.csv` or `.pkl` outputs. The scripts that produced Fig. 3 to Fig. 8 (titles such as "Free-rider removal: count and median post-activation delay") are not in the repository. Model initialization and training are unseeded (S10).
- **Evidence:** `recompute.out` ("any commit hash in metadata keys: []"); a search of `main` finds none of the figure titles.
- **Explanation:** The git history makes code drift during the runs unlikely, but the runs cannot be tied to a specific commit, and the figures cannot be regenerated from the repository.
- **Suggested remedy:** Log `git rev-parse HEAD` and a dirty-tree flag in each run's metadata. Select preset and dataset from command-line arguments. Commit the figure scripts. Seed torch per run.
- **Changed on current branch:** no

### F8. Smaller differences between the text and the code

- **Type:** paper hole
- **Status:** Confirmed
- **Severity:** Low
- **Affected claims:** M1, M3, M11, M12, R5, R9
- **Details:**
  - **Report encoding (M12):** reports are scaled by 10^4 (accuracy) and 10^2 (loss) with a uint16 cap of 65535, not 10^18.
  - **Score submission (M11):** every merged participant submits its own score; the paper says one participant computes and submits all of them.
  - **Vote threshold (M1):** absolute percentage points against the evaluator's own local accuracy, not relative to a previous model.
  - **Aggregation (M3):** unweighted mean (equivalent here), and everyone is merged when all RRS are negative.
  - **Removal delay (R5):** the delay is removal round minus activation round. A free-rider activating in round 3 with delay 6.5 actually submits free-riding updates for about 7.5 rounds and is rewarded in all but the last.
  - **Equal-share fallback:** with three or fewer merged participants, every strategy silently gives equal shares (`contribution.py:33-40`). This is rare in these runs: 51 of 2,100 Accuracy+Loss rounds and 6 of 2,100 Loss rounds have all-equal scores, which also includes the zero-sum fallback.
  - **Fig. 5 wording (R9):** under Naive on MNIST, the free-rider's GRS ends at 0.76 ETH, not "preserved", and it does not grow "like that of honest participants" (1.25 ETH).
- **Suggested remedy:** Correct the wording in §2.3, §3.2, §3.3 and §4.3 to §5.
- **Changed on current branch:** no

## Part 3: Dismissed candidates

- **Code changed between runs:** no commit on any branch touches `src/`, `contracts/` or `experiment/` between 2026-05-22 and 2026-06-02; the last code commit landed four minutes before the first run.
- **Missing or unreported runs:** exactly 10 runs per cell; `data/runs/` contains nothing outside the reported grid.
- **Run metadata disagreeing with filenames:** 0 of 480 runs.
- **Free-rider activation off by one:** the free-rider's behavior switches exactly at the configured round in 480/480 runs.
- **Missing loss negation in Accuracy+Loss** (`contribution.py:190` passes raw loss differences): signed-sum normalization is invariant to a global sign flip; the code matches Eq. 10 in 2049/2049 rounds.
- **Unweighted vs size-weighted FedAvg:** partitions are equal in size, so the two are identical.
- **Loss computed as a sum of batch means rather than a mean:** validation sets have the same size within a dataset, so comparisons between participants are unaffected.
- **uint16 loss cap distorting scores:** reached in 18 of 328,714 votes.
- **Train/test leakage:** peers evaluate on validation splits carved from each participant's training partition; global accuracy uses the separate test set; the sets are disjoint.
- **Scores not summing to one:** verified to within 2.3 × 10^-13.
- **Hardware and gas claims:** the GPUs and every gas percentage match the runs.
- **Strategies run under different conditions:** all four strategies share the same grid, run_ids and data seeds, configuration and hardware within each dataset/noise group. Model initialization differs between runs (S10), but equally across strategies.

## Part 4: Proposed next steps

### Recommendation

Fix the code, then rerun the full grid before resubmitting. F1 is not confined to the malicious-actor results: an active malicious participant votes −1 for every honest participant and submits poisoned weights. That changes RRS, merge sets, GRS and scores for everyone, including the free-rider results. Patching only the malicious arm would leave the paper's other numbers from a different setting.

If there is no compute time before the deadline, the minimum honest fallback is a text-only revision. Reframe the evaluation as "one free-rider among five (MNIST) or seven (CIFAR-10) honest participants", drop the RQ1a results, report the corrected honest-removal rates (R8), and apply the wording fixes below. The free-rider results remain valid for that setting.

### Step 1: Fix the code (no reruns yet)

Write a failing test first for each item.

1. **F1:** resolve malicious parameters in `experiments.py`, and make `PytorchModel` raise an error when a present role has no activation round. `audit/test_audit_repro.py` already contains suitable failing tests.
2. **F4:** copy the global model into every participant before round 1, and seed torch per run_id.
3. **F7:** log the commit hash and dirty-tree flag per run; take preset and dataset from command-line arguments.
4. **F2 and F3:** decide whether the code should follow the paper (Eq. 7 for all strategies; Eq. 11 without the \|c\| > 3 rule) or the paper should describe the code. This is a design decision for the authors. Aligning the code with the paper gives a cleaner story, and the paper itself already proposes a positive-only denominator as future work.

### Step 2: Smoke-test before the full rerun

Run one MNIST configuration per strategy for a few rounds and check:

- the malicious participant's behavior becomes `bad` in its start round;
- its votes are −1 and its attack type is recorded;
- round-1 global accuracy is close to the participants' local accuracy;
- the commit hash appears in the metadata.

### Step 3: Full rerun (only after your approval)

Estimated from the stored run timings:

- **MNIST, 240 runs:** about 13.5 hours on a Tesla T4.
- **CIFAR-10, 240 runs:** about 116 hours on a Tesla T4, or about 39 hours on the RX 7900 XTX.

Run all of CIFAR-10 on one GPU type so the two noise groups become comparable in wall-clock time; this also removes the hardware caveat from the Limitations section. Choose and report the malicious noise scale before starting.

### Step 4: Regenerate and revise the paper

1. Commit the figure scripts and regenerate Fig. 3 to Fig. 8 from the new runs.
2. Rerun this audit's scripts on the new data.
3. Revise the text:
   - §3.2.2: report τ and the MAD = 0 rule (F5).
   - §3.2.2 and §3.3: the normalization and settlement actually used (F2, F3).
   - §3.4: initialization and seeding (F4, S10).
   - §5: per-dataset statistics for Loss vs Accuracy (F6).
   - The smaller wording fixes in F8.
