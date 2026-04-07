from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import RunData

UUID_WITH_BRACES_REGEX = re.compile(
    r'^\{'
    r'[0-9a-fA-F]{8}-'
    r'[0-9a-fA-F]{4}-'
    r'[1-5][0-9a-fA-F]{3}-'
    r'[89abAB][0-9a-fA-F]{3}-'
    r'[0-9a-fA-F]{12}'
    r'\}$'
)

def is_valid_uuid(guid: str) -> bool:
    return bool(UUID_WITH_BRACES_REGEX.match(guid))

def extract_uuids_from_filenames(filenames: list[str] | list[RunData]) -> list[str]:
    if filenames is None:
        raise ValueError("Filenames list cannot be None")
    if not isinstance(filenames, list):
        raise ValueError("Filenames must be a list of strings or RunData objects")

    uuids = []
    for item in filenames:
        # Accept RunData objects by extracting the experiment_id (filename stem)
        filename = item.experiment_id if hasattr(item, "experiment_id") else item
        try:
            guid = extract_uuid_from_filename(filename)
            uuids.append(guid)
        except ValueError as e:
            print(f"Error processing '{filename}': {e}")
            continue

    return uuids

def extract_uuid_from_filename(filename: str):
    if filename is None:
        raise ValueError("Filename cannot be None")
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")

    idx_guid_start = filename.find('{')
    idx_guid_end = filename.find('}')

    if idx_guid_start == -1 or idx_guid_end == -1:
        raise ValueError("Invalid guid format: missing '{' or '}' in filename")

    if idx_guid_end < idx_guid_start:
        raise ValueError("Invalid guid format: misplaced braces")

    guid = filename[idx_guid_start:idx_guid_end + 1]

    if not is_valid_uuid(guid):
            raise ValueError("Invalid guid format: expected a valid UUID with braces in the filename")

    return guid