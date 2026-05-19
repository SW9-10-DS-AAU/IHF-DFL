import pytest

from analysis.uuid_extractor import extract_uuid_from_filename


VALID_GUID = "{1c20817d-ec39-445a-b845-0c68242d263e}"


# --- Happy path ---

def test_valid_guid_csv():
    filename = f"file-{VALID_GUID}.csv"
    assert extract_uuid_from_filename(filename) == VALID_GUID


def test_valid_guid_pkl():
    filename = f"file-{VALID_GUID}.pkl"
    assert extract_uuid_from_filename(filename) == VALID_GUID


def test_guid_in_middle():
    filename = f"abc-{VALID_GUID}-xyz.txt"
    assert extract_uuid_from_filename(filename) == VALID_GUID


# --- Input validation ---

def test_none_filename():
    with pytest.raises(ValueError, match="cannot be None"):
        extract_uuid_from_filename(None)


def test_non_string_filename():
    with pytest.raises(ValueError, match="must be a string"):
        extract_uuid_from_filename(123)


# --- Missing braces ---

def test_missing_opening_brace():
    filename = "file-1c20817d-ec39-445a-b845-0c68242d263e}.csv"
    with pytest.raises(ValueError, match="missing"):
        extract_uuid_from_filename(filename)


def test_missing_closing_brace():
    filename = "file-{1c20817d-ec39-445a-b845-0c68242d263e.csv"
    with pytest.raises(ValueError, match="missing"):
        extract_uuid_from_filename(filename)


def test_missing_both_braces():
    filename = "file-1c20817d-ec39-445a-b845-0c68242d263e.csv"
    with pytest.raises(ValueError, match="missing"):
        extract_uuid_from_filename(filename)


# --- Misplaced braces ---

def test_misplaced_braces():
    filename = "file-}abc{.csv"
    with pytest.raises(ValueError, match="misplaced"):
        extract_uuid_from_filename(filename)


# --- Invalid UUID format ---

def test_invalid_uuid_too_short():
    filename = "file-{1234}.csv"
    with pytest.raises(ValueError, match="valid UUID"):
        extract_uuid_from_filename(filename)


def test_invalid_uuid_wrong_format():
    filename = "file-{not-a-valid-uuid}.csv"
    with pytest.raises(ValueError, match="valid UUID"):
        extract_uuid_from_filename(filename)


def test_invalid_uuid_wrong_version():
    # version digit should be 1-5, here it's 9
    filename = "file-{1c20817d-ec39-945a-b845-0c68242d263e}.csv"
    with pytest.raises(ValueError, match="valid UUID"):
        extract_uuid_from_filename(filename)


# --- Edge cases ---

def test_multiple_brace_pairs_takes_first():
    filename = f"{VALID_GUID}-something-{VALID_GUID}.csv"
    result = extract_uuid_from_filename(filename)
    assert result == VALID_GUID  # first pair is used


def test_empty_string():
    with pytest.raises(ValueError):
        extract_uuid_from_filename("")


def test_only_braces_no_content():
    filename = "file-{}.csv"
    with pytest.raises(ValueError, match="valid UUID"):
        extract_uuid_from_filename(filename)