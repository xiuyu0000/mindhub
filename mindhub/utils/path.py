"""Utility of file path"""
import os
import json
import pathlib
from typing import Dict


__all__ = [
    "check_file_exist",
    "check_dir_exist",
    "save_json_file",
    "load_json_file",
    "detect_file_type",
]

FILE_TYPE_ALIASES = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}

ARCHIVE_TYPE_SUFFIX = [
    ".tar",
    ".zip",
]

COMPRESS_TYPE_SUFFIX = [
    ".bz2",
    ".gz",
]


def check_file_exist(file_name: str):
    """Check whether the input filename exists or not."""
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File `{file_name}` does not exist.")


def check_dir_exist(dir_name: str) -> None:
    """Check whether the input directory exist or not."""
    if not os.path.isdir(dir_name):
        raise FileNotFoundError(f"Directory `{dir_name}` does not exist.")


def save_json_file(filename: str, data: Dict) -> None:
    """Save json file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        print("Json file dump success.")


def load_json_file(filename: str) -> Dict:
    """Load json file."""
    with open(filename, "r") as f:
        res = json.load(f)
    return res


def detect_file_type(filename: str):  # pylint: disable=inconsistent-return-statements
    """Detect file type by suffixes and return tuple(suffix, archive_type, compression)."""
    suffixes = pathlib.Path(filename).suffixes
    if not suffixes:
        raise RuntimeError(f"File `{filename}` has no suffixes that could be used to detect.")
    suffix = suffixes[-1]

    # Check if the suffix is a known alias.
    if suffix in FILE_TYPE_ALIASES:
        return suffix, FILE_TYPE_ALIASES[suffix][0], FILE_TYPE_ALIASES[suffix][1]

    # Check if the suffix is an archive type.
    if suffix in ARCHIVE_TYPE_SUFFIX:
        return suffix, suffix, None

    # Check if the suffix is a compression.
    if suffix in COMPRESS_TYPE_SUFFIX:
        # Check for suffix hierarchy.
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]
            # Check if the suffix2 is an archive type.
            if suffix2 in ARCHIVE_TYPE_SUFFIX:
                return suffix2 + suffix, suffix2, suffix
        return suffix, None, suffix
