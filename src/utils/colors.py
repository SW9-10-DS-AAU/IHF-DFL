from termcolor import colored

def green(text):
    return colored(text, "green")

def gb(string):
    return colored(string, color="green", attrs=["bold"])

def rb(string):
    return colored(string, color="red", attrs=["bold"])

def b(string):
    return colored(string, color=None, attrs=["bold"])

def red(text):
    return colored(text, "red")

def yellow(text):
    return colored(text, "yellow", attrs=["bold"])
