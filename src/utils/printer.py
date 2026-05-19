import yaml
import sys
from utils.config import get_print_config


config = get_print_config()

def _print(string, end= ""):
    if config.ONLY_PRINT_ROUND_SUMMARY:
        try:
            print(string.split(":")[0]+ string.split(":")[1].split("|")[0] +
                  "                                                              ", end = "\r")
        except:
            pass
        return
    print(string, end=end)

# Horizontal divider line — replaces hard-coded "-----..." / "=====..." strings.
def print_divider(char: str = "-", blank_line_after: bool = False):
    print(char * 83)
    if blank_line_after:
        print()


def print_bar(i, l):
        p = "-" * (i+1)
        r = "." *((l-1)-i)
        _print("{}{}".format(p, r), end="\r")