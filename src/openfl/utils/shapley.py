import math

def check_shapley_compliance(processed_diffs, final_scores, tolerance=1e-7):
    violations = []
    if not processed_diffs:
        return True, []

    # 1. Efficiency:
    total_score = sum(final_scores)
    if not math.isclose(total_score, 1.0, rel_tol=tolerance):
        violations.append(f"Efficiency Broken: Sum is {total_score:.4f}")

    max_diff = max(processed_diffs)

    for i, d in enumerate(processed_diffs):
        if math.isclose(d, 0, abs_tol=tolerance):
            if not math.isclose(final_scores[i], 0, abs_tol=tolerance):
                if max_diff > tolerance:
                    violations.append(
                        f"Null Player Violation: Index {i} has 0 contribution but score {final_scores[i]:.4f}")

    pos_indices = [i for i, d in enumerate(processed_diffs) if d > tolerance]
    neg_indices = [i for i, d in enumerate(processed_diffs) if d < -tolerance]

    # 3. Symmetry
    for i in range(len(processed_diffs)):
        for j in range(i + 1, len(processed_diffs)):
            if math.isclose(processed_diffs[i], processed_diffs[j], rel_tol=tolerance):
                if not math.isclose(final_scores[i], final_scores[j], rel_tol=tolerance):
                    violations.append(f"Symmetry Violation: Index {i} and {j} have same Contribution but different scores ({final_scores[i]:.4f} vs {final_scores[j]:.4f})")

    # 4. Linearity / Proportionality
    for group in [pos_indices, neg_indices]:
        for idx_a in range(len(group)):
            for idx_b in range(idx_a + 1, len(group)):
                i, j = group[idx_a], group[idx_b]

                if abs(processed_diffs[j]) > tolerance and abs(final_scores[j]) > tolerance:
                    in_ratio = processed_diffs[i] / processed_diffs[j]
                    out_ratio = final_scores[i] / final_scores[j]

                    is_linear = math.isclose(in_ratio, out_ratio, rel_tol=1e-7)
                    is_inverted = math.isclose(in_ratio, 1 / out_ratio, rel_tol=0.1) if out_ratio != 0 else False

                    if not (is_linear or is_inverted):
                                violations.append(
                                    f"Linearity Violation: Inconsistent scaling between index {i} and {j}. Expected ~{in_ratio:.2f}, but got {out_ratio:.2f}.")

    return len(violations) == 0, violations