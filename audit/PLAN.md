# Plan: fix the code before rerunning the FAIRFL experiments

Date: 2026-10-27. Branch: `BugfixOctober` (same commit as `main`, `d702bb6`, when the plan was made).
Tracking: GitHub parent issue #81, sub-issues #82–#89, triage issues #90–#95.

## How to review this plan

This plan was made in a decision session with the authors, based on an audit of the paper against the code and the stored runs. Please check:

1. **Correctness of the diagnosis.** Does each "Problem" below match the code at the cited lines?
2. **Correctness of the fix.** Does each fix solve the problem without breaking something else (other presets, `experiment/sample.py`, the analysis layer, the contract's accounting)?
3. **Consistency with the paper.** The goal is that the code does what the paper (`polfl.pdf`, §3.2–§3.4) says, except where this plan explicitly keeps the code and changes the text instead.
4. **Gaps.** Anything that would make the 480-run rerun invalid, or that a reviewer of the paper would catch, and that this plan does not cover.
5. **The risks in the last section.** Especially the counterfactual result on free-rider removal.

Background documents in this folder:
- `REPORT.md`: the audit, with finding IDs F1–F8 and claim IDs (S, M, R).
- `NOTES.md`: decisions, deferred items and the paper-text list.
- Scripts and outputs (`recompute.py`, `check_scoring.py`, `misc_checks.py`, `counterfactual.py` and their `.out` files) to reproduce the audit's numbers. They need `audit/cache.pkl`, built by `load_all.py` from `data/runs/`.

## Context in five lines

- The paper evaluates FAIRFL, which adds contribution scoring (Naive, Loss, Accuracy, Accuracy+Loss) to the OpenFL voting scheme, on MNIST and CIFAR-10: 4 strategies × 2 free-rider noise levels × 3 free-rider activation rounds × 10 repetitions = 480 runs.
- Every number printed in the paper reproduces from the stored runs.
- But the malicious participant never activated (F1), Loss and Accuracy did not use the paper's Eq. 7 (F2), and the contract's settlement did not follow Eq. 11 (F3).
- Decision: fix the code and rerun all 480 runs (about 13.5 h MNIST, about 39 h CIFAR-10).
- A partial rerun of only the malicious arm is not valid, because an active malicious participant changes votes, merge sets, GRS and scores for everyone.

## Decisions

| Topic | Decision | Rejected alternative and why |
|---|---|---|
| Rerun or text-only | Fix the code and rerun all 480 runs | Text-only revision: would keep results with no poisoning at all |
| Malicious start round | Round 1 in the four paper presets, as §3.4 says | Copying the free-rider's round (1, 3 or 5): contradicts §3.4 and mixes up the two roles' timing |
| Malicious noise scale | The run's free-rider noise (0.01 or 0.1) | A fixed value such as 0.1: the paper names no value; the authors chose to tie it to Table 4's noise |
| Normalization | Eq. 7 (signed sum) for Loss, Accuracy and Accuracy+Loss | Clamped `_new` for all (proposed by a second review): contradicts the paper; positive-only denominator: the paper leaves it as future work |
| Settlement | Eq. 11 in the contract, as the paper describes | Keep the contract and describe it: its instant removal at \|c\| > 3 would hit all three metric strategies under Eq. 7 |
| Filter threshold τ | Keep 1.1, report it | 3.0 (Leys et al.): the paper states no value, so no change |
| Shared initial model, seeding | Deferred (#94) | — |
| Small text/code differences (F8) | Keep the code, fix the paper text | — |
| Dead code (`normalize_contribution_scores_new`) | Keep for now | — |

## Work items, in order

Each code change starts with a test that fails because of the bug. Commits stay local; the authors push.

### 1. Rebuild the paper's figures (#82)

**Problem:** the scripts behind Fig. 3–8 are not in the repository.
**Change:** new `analysis/paper_figures.py`, which reads `data/runs/` through `analysis/loader.py` and `analysis/transform.py` and writes Fig. 3–8 to `data/output_graphs/paper/`. The bootstrap seed is fixed.
**Acceptance:** on the existing 480 runs, every Fig. 3 cell and the Fig. 4, 6, 7 and 8 numbers match the paper (exact targets are in #82). This also gives a baseline to compare against after the later changes.
**Why first:** it proves the analysis pipeline reproduces the paper before anything changes.

### 2. Malicious participant activates (F1, #83)

**Problem:** the four paper presets set `malicious_start_round=None` and `malicious_noise_scale=None`. `experiment/experiments.py:217-218` writes these `None` values over the fallback that `ExperimentConfiguration.__init__` (`experiment/experiment_configuration.py:45-46`) had set. `src/ml/attacks.py:90` switches attitude only when `attitudeSwitch == round`, which is never true for `None`. Result: role `bad` behaved `good` in 8,880 of 8,880 participant-rounds.
**Change:**
- `experiments.py:217-218`: fall back to this run's free-rider round and noise when the preset value is `None`.
- The four paper presets: `malicious_start_round=[1]`; `malicious_noise_scale` stays `None` (so it takes the free-rider noise).
- `experiment/experiment_runner.py`, before `PytorchModel` is built: raise `ValueError` when bad contributors exist and either value is `None`.
- Check that skip detection (`experiments.py` around 313-314, `resolve_attack_params` in `experiment/helper.py`) still matches the new filenames.
**Acceptance:** per-run configuration for every paper preset cell has start round 1 and the run's free-rider noise; a `bad` participant switches in round 1; the runner raises on `None`; `experiment/sample.py` still works.

### 3. Eq. 7 for Loss and Accuracy (F2, #84)

**Problem:** Accuracy (`src/contracts/contribution.py:254`) and Loss (`:313`) call `normalize_contribution_scores_new`, which clamps negatives at −1. Only Accuracy+Loss uses Eq. 7 (`normalize_contribution_scores_old`, `:187-190`).
**Change:** both call `normalize_contribution_scores_old`.
**Acceptance:** improvements [1, 2, −10, −5] give [−1/12, −2/12, 10/12, 5/12]; scores sum to 1. Existing tests that check `_new` behavior through these strategies are updated to the Eq. 7 expectation and listed in the PR.
**Note:** Eq. 7's sign inversion when improvements sum to a negative number is intended; the paper describes it.

### 4. Settlement by Eq. 11 (F3, #86)

**Problem:** `contracts/OpenFLModel.sol:529-552` charges a negative score `GRS·|c|/3` and disqualifies at once when that exceeds GRS (equivalent to |c| > 3). The paper's Eq. 11 is `GRS_i ← GRS_i + R·w_i / Σ_k max(w_k, 0)` with `w_i = v_i·c_i`. 72 of the 88 Accuracy+Loss removals of non-free-riders in the old runs came from the |c| > 3 rule.
**Change:**
- Negative `w_i`: lose `R·|w_i| / Σ positive w`. Remove the `GRS·|c|/3` penalty and the `punishment > GRS` disqualification.
- If the loss is at least the GRS: GRS becomes 0 and the participant is removed (no underflow revert).
- Removal at `GRS ≤ min_collateral / punishfactorContrib` stays.
- Penalties and stakes forfeited in this settlement go into the **next** round's reward, computed after all penalties for this round (the penalty depends on R, so adding it to this round's R would be circular).
- Update any Python code in `src/contracts/fl_challenge.py` that mirrors the settlement.
**Acceptance:** Foundry tests in `tests/solidity/` for each rule above; `forge test` passes in WSL (`~/.foundry/bin/forge` in Ubuntu-24.04); the Python suite passes.
**Paper gap:** the clamp and where lost GRS goes are not in the paper; both are on the paper-text list.

### 5. CIFAR-10 validation without augmentation (#87)

**Problem:** `src/ml/data.py:14-17` gives the CIFAR-10 training set `RandomCrop` and `RandomHorizontalFlip`. The validation split (`:62`) comes from the same dataset, so peer evaluation (votes and the accuracy/loss reports) runs on randomly augmented images, which differ on every evaluation. Present since the first commit.
**Change:** build validation subsets from an unaugmented copy of the training set, with the same indices (the subsets are nested, so map the indices through each level).
**Acceptance:** the same validation sample loads identically twice; validation indices per run_id are unchanged; training batches are still augmented.

### 6. Traceable runs (F7, #85)

**Problem:** `experiment/experiments.py:38-40` hard-codes the preset and dataset, so paper runs needed uncommitted edits; no commit hash is stored.
**Change:** `--preset`, `--dataset` and `--allow-dirty` flags (defaults keep today's behavior); `git_commit` and `git_dirty` (tracked files only) in each run's metadata; refuse a dirty tree or missing git unless `--allow-dirty`; update `README.md:64-71`.
**Acceptance:** metadata has both keys; dirty tree stops the run with a message naming the flag; figures from step 1 are unchanged before and after (`analysis/transform.py:51` keeps only listed metadata keys).

### 7. Smoke test (#88, blocked by #83–#87)

One short MNIST run per strategy (noise 0.1, activation round 3) and one short CIFAR-10 run, on a clean tree. Check: malicious behavior `bad` from round 1 with votes −1 and `attack_type` `noise`; its noise equals the free-rider's; Loss and Accuracy scores follow Eq. 7; no removal by the old |c| > 3 rule; CIFAR-10 validation accuracy of an unchanged model is stable; `git_commit` and `git_dirty=False` in the metadata; `paper_figures.py` runs on the output.

### 8. Full rerun, then paper revision (#89)

After the GPU choice (#95): rerun 480 runs, regenerate Fig. 3–8, rerun the audit scripts on the new data, and revise the text per #89.

## Not in this plan

- Shared initial model before round 1 and torch seeding (#94). The paper's §3.4 must then state that round 1 averages independently initialized models and that only the data split is seeded.
- Second-review findings that do not affect paper figures: `dotproduct` MAD trim is a no-op (#90), per-role accuracy columns (#91), round-0 zeros in a relative-round aggregation (#92), missing votes for a participant disqualified in that round (#93).
- Second-review findings rejected: switching Accuracy+Loss to the clamped normalization (contradicts the paper; the sign inversion is Eq. 7's documented behavior, and the missing loss negation is harmless because n(d) = n(−d)); loss as a sum of batch means (the paper never compares loss across datasets, and validation sets within a dataset are the same size).

## Risks and open questions

1. **Free-rider removal may weaken under Eq. 7 + Eq. 11.** `counterfactual.py` replays the old runs' settlement under both rule sets, holding merge sets and filtered reports fixed. The current-rule replay reproduces 100% of the observed removals. Under the paper's rules, low-noise free-riders removed on CIFAR-10 drop from 30/30 to 10/30 for Loss and to 8/30 for Accuracy, with the rest "censored" (the real run removed them earlier, so the replay has no later data). On MNIST: Loss 29 → 11 (+17 censored), Accuracy 21 → 11 (+8 censored). Honest removals mostly become censored too (CIFAR-10 Accuracy+Loss 65 → 0 removed, 65 censored). The replay cannot say how the censored cases end, but the paper's headline result (Loss removes 59/60 low-noise free-riders) may not hold after the rerun.
2. **At noise 0.01 the malicious participant barely poisons.** In the old runs, noise 0.01 lowered the free-rider's accuracy by about 3 percentage points, below the 7-point vote threshold. Its update will usually be merged, so half the grid tests mainly the malicious votes and reports, not poisoned weights.
3. **The malicious participant's reports now reach the filter.** It reports accuracy 0 and loss 65535 for everyone. With τ = 1.1 they will be filtered, but so are about 27% of honest reports.
4. **Equal-share fallback becomes more frequent.** With three or fewer merged participants every strategy gives equal shares. On MNIST, only four participants remain once both attackers are removed.
5. **Eq. 11 details beyond the paper** (clamp at zero; lost GRS to the next round). Another reasonable reading: lost GRS is simply burned. Check that the chosen rule keeps the contract's ETH accounting balanced.
6. **GPU choice (#95)** is open and blocks the rerun.
