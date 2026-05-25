"""Back-compat re-export shim — relocated to cinemateca.annotations.io.

Existing import paths ``from cinemateca.annotator import load, save,
merge_tag_index`` keep working through this shim. P3.A Task 6 deletes
this file after every importer migrates.
"""
from cinemateca.annotations.io import FILENAME, load, merge_tag_index, save  # noqa: F401

__all__ = ["FILENAME", "load", "merge_tag_index", "save"]
