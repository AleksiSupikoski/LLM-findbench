import os
from difflib import SequenceMatcher
from typing import List, Tuple, Union


def read_context(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def normalize_path(path: str) -> str:
    return os.path.normpath(path).lower()


def parse_lines(lines_field: Union[str, List[str], None]) -> List[Tuple[int, int]]:
    """
    Parse line ranges, supporting multiple ranges separated by semicolon.
    Returns list of (start_line, end_line) tuples.
    """
    ranges = []
    if lines_field is None:
        return ranges
    if isinstance(lines_field, str):
        parts = lines_field.split(";")
        for part in parts:
            try:
                a, b = map(int, part.strip().split("-"))
                ranges.append((a, b))
            except Exception:
                continue
    elif isinstance(lines_field, list):
        for r in lines_field:
            try:
                a, b = map(int, r.strip().split("-"))
                ranges.append((a, b))
            except Exception:
                continue
    return ranges


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping or contiguous ranges.
    Input must be list of tuples (start, end).
    Returns list of merged ranges sorted by start.
    """
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:  # Overlapping or contiguous
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def iou(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
    """
    Compute Intersection over Union for two line ranges.
    """
    s1, e1 = range1
    s2, e2 = range2
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union if union > 0 else 0.0


def compute_iou_between_ranges_list(ranges1: List[Tuple[int, int]], ranges2: List[Tuple[int, int]]) -> float:
    """
    Compute IoU between two sets of line ranges.
    Approach: sum intersection length / sum union length.
    """
    if not ranges1 or not ranges2:
        return 0.0

    # Merge ranges to avoid overlaps internally
    r1 = merge_ranges(ranges1)
    r2 = merge_ranges(ranges2)

    # Compute intersection length
    inter_len = 0
    for a_start, a_end in r1:
        for b_start, b_end in r2:
            inter_start = max(a_start, b_start)
            inter_end = min(a_end, b_end)
            if inter_start <= inter_end:
                inter_len += (inter_end - inter_start + 1)

    # Compute union length
    def total_length(ranges):
        return sum(e - s + 1 for s, e in ranges)

    union_len = total_length(r1) + total_length(r2) - inter_len
    if union_len == 0:
        return 0.0
    return inter_len / union_len

def fuzzy_match_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()