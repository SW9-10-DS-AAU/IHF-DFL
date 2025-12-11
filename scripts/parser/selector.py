def _parse_selection(expr: str, max_index: int, single: bool) -> set[int]:
    """
    Supports:
      - "1 2 3"
      - "1-4"
      - "1 3-5 7"
      - "^4"   (exclude)
      - combined: "1-10 ^3 ^7"

    If single=True:
      - Only a single index is allowed
      - Ranges, exclusions, or multiple tokens are invalid
    """
    tokens = expr.split()

    if single:
        # Must be exactly one token
        if len(tokens) != 1:
            raise ValueError("Only one selection is allowed.")

        token = tokens[0]

        # Ranges not allowed
        if "-" in token or token.startswith("^"):
            raise ValueError("Only a single index is allowed.")

        idx = int(token)
        if not (0 <= idx <= max_index):
            raise ValueError("Index out of range.")

        return {idx}

    # Multi-selection mode below

    selected = set()

    # Handle positive selections
    for token in tokens:
        if token.startswith("^"):
            continue  # exclusions handled later

        if "-" in token:
            start, end = token.split("-")
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(token))

    # Apply exclusions
    for token in tokens:
        if token.startswith("^"):
            idx = int(token[1:])
            selected.discard(idx)

    # Bound check
    return {i for i in selected if 0 <= i <= max_index}


def choose_from_list(options: list[str], question: str, single: bool = False) -> list[str]:
    if not options:
        return []

    print(question + "\n")

    for i, item in enumerate(options):
        print(f"[{i}] {item}")

    if (single):
      print('\nSelect')
    else:
      print('\nSelect using "1 2 3", "1-3", "^2", or combinations like "1-5 ^3".')
    raw = input("> ").strip()

    indices = _parse_selection(raw, len(options) - 1, single)

    return [options[i] for i in indices]