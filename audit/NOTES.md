# Notes from the audit grilling session (2026-10-27)

Decisions taken after `REPORT.md`, and what the paper must say once the rerun is done. Tracked on GitHub under parent issue #81 (sub-issues #82–#89) and triage issues #90–#95. Finding IDs (F1–F8, M1–M13, R1–R25) refer to `REPORT.md`. Review IDs (A1–C) refer to the second agent's `RESULTS_BUG_REVIEW.md`.

## Decided code changes before the rerun

- **F1:** the malicious participant starts in round 1 in the four paper presets. Its noise scale falls back to the run's free-rider noise (0.01 or 0.1). Runs fail fast when a role is present but has no start round or noise scale.
- **F2:** Loss, Accuracy and Accuracy+Loss all use Eq. 7 (`normalize_contribution_scores_old`). `normalize_contribution_scores_new` stays in the code, unused, for now.
- **F3:** the contract settles negative scores by Eq. 11. The GRS·|c|/3 penalty and the instant removal at |c| > 3 go away.
- **F7:** preset and dataset come from command-line flags; each run records `git_commit` and `git_dirty`; a dirty tree is refused unless `--allow-dirty` is passed.
- **CIFAR validation (review C):** peer evaluation uses unaugmented images. The bug dates from the first commit (`c9e9ebc`, from the original OpenFL code).
- **Figures:** Fig. 3–8 are rebuilt by a script in `analysis/`, because the original scripts are not available.
- A smoke test runs before the full rerun.

## Decide before the rerun

- **GPU choice (old Q9):** all CIFAR-10 on the RX 7900 XTX (about 39 h) and MNIST on the T4 (about 13.5 h) was the recommendation. Not decided yet.

## Deferred code changes (not before this rerun unless decided otherwise)

- **F4, shared initial model:** participants start from their own random networks; the global model is sent out only after the first merge. This has been so since the first commit (`c9e9ebc`). Fix: copy the global model into every participant before round 1.
- **Seeding:** only the data split is seeded. Model init, DataLoader shuffling and attack noise are not. Fix: seed torch per run_id, without full GPU determinism.
- **τ = 1.1** stays, because the paper gives no value.
- `normalize_contribution_scores_new` and the unused `_freerider_submit_with_noise` and `add_noise` stay as dead code for now.
- Review A3 (dotproduct MAD trim is a no-op), B2 (per-role accuracy columns), B3 (round-0 zeros in the relative contribution plot) and B5 (votes of a user disqualified in that round are missing from the log) are filed as `needs-triage` issues. None of them affects a paper figure.

## Paper text to revise after the rerun

### §2.3, §3.2, §3.3 (method)

- **Vote threshold (M1):** the 7% and 14% thresholds are absolute percentage points, compared with the evaluator's own local validation accuracy, not with a previous model.
- **Aggregation (M3):** an unweighted mean (equal to FedAvg here, because partitions are equal in size). When every RRS is negative, every participant is merged.
- **Filter (F5, M5):** report τ = 1.1. The code's MAD = 0 rule (keep reports within τ raw units of the median) equals Eq. 4 with ε = 0.6745 raw report units; the only difference is that the code divides by MAD = 0.5 where Eq. 4 would divide by ε. Note that τ = 1.1 is far below the 2.5–3.5 suggested by Leys et al. and discarded about 27% of honest reports in the old runs.
- **Normalization (F2):** §3.2.2 is now correct for all three metric strategies. Consider adding the concrete cost of Eq. 7: for accuracies [8, 9, 11] against a baseline of 10, the only participant that improved gets −0.5 and the worst gets 1.0.
- **Settlement (F3):** Eq. 11 is now implemented. State the two details the paper leaves open: GRS is clamped at zero (the participant is then removed), and the GRS lost through a negative score is added to the next round's reward.
- **Score submission (M11):** every merged participant submits its own score in its own transaction; the paper says a single participant submits them all.
- **Report encoding (M12):** accuracy reports are scaled by 10^4 and losses by 10^2, capped at 65535 (uint16). Only contribution scores use 10^18.
- **Equal-share fallback (A4):** with three or fewer merged participants every strategy gives equal shares. Expect this more often after the rerun: on MNIST only four participants remain once both attackers are removed.

### §3.4 (experimental setting)

- **Malicious participant:** starts in round 1 and uses the run's free-rider noise (0.01 or 0.1). Like the free-rider, it does not train: it adds noise to its current model. At noise 0.01 its update lowers accuracy by only about 3 percentage points and passes the vote, so "high loss/low accuracy, honest actors assign negative votes" holds only at 0.1. It votes −1 for every other participant, the free-rider included (the paper says "all honest participants"), and reports accuracy 0 and loss 65535.
- **Initialization (F4):** unless F4 is fixed, state that round 1 averages independently initialized models, and that a free-rider activating in round 1 submits its own random network plus noise.
- **Seeding (S10):** only the data split is seeded.
- **Hardware:** update the sentence once the GPU choice is made.

### §4 and §5 (results and discussion)

- **RQ1a:** rewrite from the new runs; the old text describes an attack that never ran.
- **Honest-removal rates (R8):** in the old runs the "malicious" participant was honest, so the honest rates were under-reported (MNIST 23/300, 1/300, 1/300, 0/300; CIFAR-10 65/420, 54/420, 57/420, 0/420). Use the new runs.
- **Removal delay (R5):** the delay is the removal round minus the activation round; the free-rider submits free-riding updates for one round more than the delay states.
- **Loss vs Accuracy (F6):** report per-dataset counts with the test. In the old runs the gap came only from MNIST (29/30 vs 21/30, Fisher p = 0.012); CIFAR-10 was 30/30 for both.
- **Fig. 5 wording (R9):** in the old runs, the Naive free-rider on MNIST ended at 0.76 ETH, neither "preserved" nor growing like the honest participants (1.25 ETH). Recheck against the new runs.
- **Reproducibility (F7):** mention that each run records its commit, and that the figure scripts are in the repository.
