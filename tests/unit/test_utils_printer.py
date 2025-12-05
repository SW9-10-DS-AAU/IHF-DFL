from types import SimpleNamespace

import openfl.utils.printer as printer


def test_print_respects_only_summary_flag_false(capsys, monkeypatch):
    monkeypatch.setattr(printer, "config", SimpleNamespace(ONLY_PRINT_ROUND_SUMMARY=False))

    printer._print("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_print_respects_only_summary_flag_true(capsys, monkeypatch):
    monkeypatch.setattr(printer, "config", SimpleNamespace(ONLY_PRINT_ROUND_SUMMARY=True))

    printer._print("round:summary|extra")
    captured = capsys.readouterr()
    assert "round" in captured.out
    assert "summary" in captured.out


def test_print_bar_delegates_to_print(monkeypatch):
    calls = []

    def fake_print(text, end=""):
        calls.append((text, end))

    monkeypatch.setattr(printer, "_print", fake_print)
    printer.print_bar(1, 5)

    assert calls
    rendered, end = calls[0]
    assert rendered.startswith("--")
    assert rendered.endswith("...")
    assert end == "\r"
