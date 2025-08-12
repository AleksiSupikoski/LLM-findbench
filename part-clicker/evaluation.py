# evaluation.py
from typing import List, Dict, Tuple
from utils import normalize_path, parse_lines, merge_ranges, compute_iou_between_ranges_list
from collections import defaultdict
from typing import List, Dict, Tuple
from utils import normalize_path, parse_lines, iou, fuzzy_match_score


def collect_and_merge_ranges(subfeatures: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Collects subfeatures per file and merges their line ranges.
    Returns dict: file_path -> merged line ranges list.
    """
    file_to_ranges = {}

    for sf in subfeatures:
        file_path = normalize_path(sf["file_path"])
        lines_raw = sf.get("lines")
        ranges = parse_lines(lines_raw)

        if file_path in file_to_ranges:
            file_to_ranges[file_path].extend(ranges)
        else:
            file_to_ranges[file_path] = ranges

    # ----------------------------------------- Merge ranges per file
    for file_path in file_to_ranges:
        file_to_ranges[file_path] = merge_ranges(file_to_ranges[file_path])

    return file_to_ranges


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or contiguous line ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:  # -------------------- overlap or contiguous
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def collect_file_ranges(subfeatures: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Group and merge line ranges by normalized file path.
    """

    grouped = defaultdict(list)
    for sf in subfeatures:
        file_path = normalize_path(sf["file_path"])
        ranges = parse_lines(sf.get("lines"))
        grouped[file_path].extend(ranges)
    return {fp: merge_ranges(rngs) for fp, rngs in grouped.items()}

def evaluate_category(pred_list: List[Dict], gold_list: List[Dict], iou_threshold=0.5, path_threshold=0.8) -> Tuple[float, float, float, float, int, int, int, int, int]:
    """
    Evaluate predictions vs gold for a category using merged IoU per file.
    Returns precision, recall, f1, avg_iou, num_gold_files, num_pred_files, TP, FP, FN.
    """

    gold_files = collect_file_ranges(gold_list)
    pred_files = collect_file_ranges(pred_list)

    matched_files = set()
    total_iou = 0.0
    iou_count = 0
    TP = 0
    FP = 0

    # Match predicted files to gold files
    for p_file, p_ranges in pred_files.items():
        best_iou = 0.0
        best_gold_file = None
        for g_file, g_ranges in gold_files.items():
            if fuzzy_match_score(p_file, g_file) < path_threshold:
                continue
            all_iou = [iou(pr, gr) for pr in p_ranges for gr in g_ranges]
            file_iou = max(all_iou) if all_iou else 0.0
            if file_iou > best_iou:
                best_iou = file_iou
                best_gold_file = g_file
        if best_iou > 0:
            TP += 1
            total_iou += best_iou
            iou_count += 1
            matched_files.add(best_gold_file)
        else:
            FP += 1

    FN = len(gold_files) - len(matched_files)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0

    return precision, recall, f1, avg_iou, len(gold_files), len(pred_files), TP, FP, FN


"""
def evaluate_category(pred_list: List[Dict], gold_list: List[Dict], iou_threshold=0.5) -> Tuple[float, float, float, float, int, int]:
    """

    #Evaluate predictions vs gold for a category by merging ranges per file and computing IoU.
    #Returns precision, recall, f1, TP (sum IoU), FP, FN.

"""

    # Collect & merge line ranges per file for gold and predicted
    gold_files = collect_and_merge_ranges(gold_list)
    pred_files = collect_and_merge_ranges(pred_list)

    TP = 0.0
    FP = 0
    FN = 0

    matched_gold_files = set()
    matched_pred_files = set()

    # Compute matches by file
    for pred_file, pred_ranges in pred_files.items():
        if pred_file in gold_files:
            gold_ranges = gold_files[pred_file]
            iou_score = compute_iou_between_ranges_list(pred_ranges, gold_ranges)
            if iou_score >= iou_threshold:
                TP += iou_score
                matched_gold_files.add(pred_file)
                matched_pred_files.add(pred_file)
            else:
                FP += 1
        else:
            FP += 1

    # Gold files not matched count as FN
    FN = len(gold_files.keys() - matched_gold_files)

    num_gold = len(gold_files)
    num_pred = len(pred_files)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, TP, FP, FN
"""