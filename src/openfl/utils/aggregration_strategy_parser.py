import re

PARTIAL_SWITCH_MODE_MAP = {
"loss": "partial_switch_fixed_loss",
"acc": "partial_switch_accuracy",
"retro": "partial_switch_retrospective",
}

def extract_values(input_str: str):
    parts = [s.strip() for s in input_str.split(',')]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Expected 2 or 3 comma-separated values, got {len(parts)}: '{input_str}'")


def parse_partial_switch(input_str: str):
    inner = input_str.removeprefix('partial_switch{').removesuffix('}')
    mode, func1, func2 = extract_values(inner)
    if mode not in PARTIAL_SWITCH_MODE_MAP:
        raise ValueError(f"Unknown partial switch mode '{mode}'. Valid: {list(PARTIAL_SWITCH_MODE_MAP)}")
    return PARTIAL_SWITCH_MODE_MAP[mode], func1, func2

def parse_binary_switch(input_str: str):
    inner = input_str.removeprefix('binary_switch{').removesuffix('}')
    *_, func1, func2 = extract_values(inner)  # leading token is an optional, unused metric hint
    return "binary_switch", func1, func2

def get_switch_type(input_str: str) -> str:
    match = re.match(r"^(\w+_switch)\{", input_str)
    return match.group(1) if match else "unknown"


def parse_values(input_str: str):
    switch_type = get_switch_type(input_str)
    if switch_type == "binary_switch":
        return parse_binary_switch(input_str)
    elif switch_type == "partial_switch":
        return parse_partial_switch(input_str)
    else:
        raise ValueError(f"Unknown switch type in input: {input_str}")


if __name__ == "__main__":
    test_str_1 = "binary_switch{acc, positives_only, plus_one_normalize}"
    res_str_1 = parse_values(test_str_1)
    test_str_2 = "partial_switch{loss, positives_only, plus_one_normalize}"
    res_str_2 = parse_values(test_str_2)

    print(f"Binary switch: type: {res_str_1[0]}, func1: {res_str_1[1]}, func2: {res_str_1[2]}")
    print(f"Partial switch: type: {res_str_2[0]}, func1: {res_str_2[1]}, func2: {res_str_2[2]}")

