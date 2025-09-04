"""
Evaluate LLMs on locating implementation points for feature requests.

Core design:
- Atomic correctness: (filepath, line-range) must match. A prediction is correct iff:
    * file paths match AND
    * predicted line is inside the gold range, OR within ±tolerance of its bounds, OR gold lines are None/"any".
- Required vs Optional:
    * Recall counts ONLY REQUIRED gold locations (OPTIONAL never creates FNs).
    * Predictions hitting OPTIONAL gold count as TPs (affect precision), not recall.

Metrics (rank-free, multi-run):
- --num-runs K: query the model K times per example.
- Per-run macro (per example): average Precision/Recall/F1/EM over K runs, then macro-average over examples.
- Union metrics (per example): Precision/Recall/F1/EM on the union of unique predictions across K runs (atomic).
- Pass@{1,3,5,k} (per example): HumanEval-style estimator using c = #runs with EM=1 (atomic, strict: FN=0 & FP=0),
  macro-averaged over examples.

Also supports interactive adjudication (--stop-on-fp) applied to the UNION of FPs (adds as OPTIONAL).

Usage:
  python script.py \
      --dataset data.json \
      --code-context gptree_output.txt \
      --model gpt-4o-mini \
      --outdir runs \
      --line-tolerance 10 \
      --max-context-chars 120000 \
      --num-runs 5 \
      [--stop-on-fp]

Requires:
  pip install python-dotenv openai (>=1.30), pandas (optional), numpy
  .env with OPENAI_API_KEY=...
"""

import argparse
import csv
import json
import os
import re
import sys
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

from dotenv import load_dotenv

# ----------- OpenAI client (API v1)
try:
    from openai import OpenAI
except Exception as e:
    print("Please install openai>=1.30: pip install --upgrade openai", file=sys.stderr)
    raise


# ---------- Helpers

def load_text(path: str, max_chars: Optional[int] = None) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if max_chars and len(txt) > max_chars:
        head = txt[: max_chars // 2]
        tail = txt[-max_chars // 2 :]
        txt = head + "\n\n...[TRUNCATED]...\n\n" + tail
    return txt


def parse_line_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Accepts "147-153", "147", "any", "unknown", "", None.
    Returns (start, end) inclusive, or None if unknown/any.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        n = int(s)
        return (n, n)
    s = str(s).strip().lower()
    if s in {"any", "unknown", "null", "n/a", ""}:
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return (a, b)
    m = re.match(r"^\s*(\d+)\s*$", s)
    if m:
        n = int(m.group(1))
        return (n, n)
    return None


def within_tolerance(pred: Optional[int], gold_range: Optional[Tuple[int, int]], tol: int) -> bool:
    """
    Atomic line-match predicate:
      - If gold_range is None -> accept any pred (including None).
      - If pred is None and gold_range exists -> mismatch.
      - Else match if pred within range or within ±tol of nearest bound.
    """
    if gold_range is None:
        return True
    if pred is None:
        return False
    start, end = gold_range
    if start <= pred <= end:
        return True
    delta = min(abs(pred - start), abs(pred - end))
    return delta <= tol


def extract_json_list(text: str) -> List[Dict[str, Any]]:
    """Pull the first top-level JSON list from the model response."""
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Fallback: bracket slice
    first = text.find('[')
    last = text.rfind(']')
    if first != -1 and last != -1 and last > first:
        snippet = text[first:last + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []


def flatten_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Emit items with:
      g_idx, f_idx, superfeature, tag, feature_desc,
      required_paths, optional_paths,
      gold_lines_map (path -> Optional[(start,end)]),
      optional_map (path -> bool)
    """
    items = []
    for g_idx, group in enumerate(dataset):
        for f_idx, feat in enumerate(group.get("features", [])):
            fps = feat.get("file_paths", []) or []
            required_paths: List[str] = []
            optional_paths: List[str] = []
            lines_map: Dict[str, Optional[Tuple[int, int]]] = {}
            optional_map: Dict[str, bool] = {}
            for fp in fps:
                p = fp.get("path")
                if not p:
                    continue
                opt = bool(fp.get("optional", False))
                rng = parse_line_range(fp.get("lines"))
                lines_map[p] = rng
                optional_map[p] = opt
                (optional_paths if opt else required_paths).append(p)
            items.append({
                "g_idx": g_idx,
                "f_idx": f_idx,
                "superfeature": group.get("superfeature"),
                "tag": feat.get("tag"),
                "feature_desc": feat.get("feature_desc"),
                "required_paths": required_paths,
                "optional_paths": optional_paths,
                "gold_lines_map": lines_map,
                "optional_map": optional_map,
            })
    return items


def compute_metrics(tp_total: int, fp: int, fn_required: int,
                    total_required_gold: int, total_pred: int,
                    tp_required: int, tp_optional: int) -> Dict[str, float]:
    """
    Precision = (TP_required + TP_optional) / total_pred
    Recall (required) = TP_required / total_required_gold
    F1 = 2 * P * R / (P + R)
    Accuracy (over required universe) = TP_total / (TP_total + FP + FN_required)
    """
    precision = tp_total / total_pred if total_pred > 0 else 0.0
    recall = tp_required / total_required_gold if total_required_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    denom = tp_total + fp + fn_required
    accuracy = tp_total / denom if denom > 0 else 0.0
    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                tp_required=tp_required, tp_optional=tp_optional)


def build_task_text(tag: str, superfeature: str, feature_desc: str) -> str:
    """Concise prompt text to log (NO code context)."""
    return f"""You are a senior software engineer assisting in feature localization. Given the repository below, locate where **{tag}** of a feature **{superfeature}**: **{feature_desc}** needs to take place. Provide the required file_path and the code line number where this change should occur. There can be multiple locations. Only if the edit can be done anywhere in a given file, set line_number to null (for example if it just can be appended code). Answer **only** with a JSON array, each element an object with keys ["file_path", "line_number"]. No commentary. 
    
    Example:
    Feature: "Add logout button in top navigation bar"
    Answer:
    [{{"file_path": "src/components/Navbar.js", "line_number": 120}}]

    Example:
    Feature: "Change app title text"
    Answer:
    [{{"file_path": "public/index.html", "line_number": 15}}]

    First, think step by step: identify likely files or modules, then narrow down to specific functions or lines.
    
    Context: """


def prompt_template(tag: str, superfeature: str, feature_desc: str, code_context: str) -> List[Dict[str, str]]:
    """Build chat messages. Model must answer with JSON list of {file_path, line_number}."""
    task = build_task_text(tag, superfeature, feature_desc)
    user_content = task + "\n\nCodebase context (gptree_output.txt excerpt):\n" + code_context
    return [
        {"role": "system", "content": "You are a precise code navigation assistant. Reply only in JSON as instructed."},
        {"role": "user", "content": user_content}
    ]


def call_openai(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"} if model.startswith("gpt-4.1") else None,
    )
    return resp.choices[0].message.content


def to_csv_safe_list(xs: List[Any]) -> str:
    return ";".join(map(str, xs))


def add_fp_to_dataset(dataset_path: str, dataset_obj: List[Dict[str, Any]],
                      item_idx: Tuple[int, int], new_path: str) -> None:
    """
    Add new OPTIONAL file path with lines='any' to dataset for the specified (group_idx, feature_idx).
    Writes back to dataset_path (overwrites).
    """
    g_idx, f_idx = item_idx
    try:
        group = dataset_obj[g_idx]
        feat = group["features"][f_idx]
        if "file_paths" not in feat or not isinstance(feat["file_paths"], list):
            feat["file_paths"] = []
        feat["file_paths"].append({
            "path": new_path,
            "lines": "any",
            "optional": True,
            "code_preview": []
        })
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to update dataset with new path '{new_path}': {e}", file=sys.stderr)


def fmt_line_range(r: Optional[Tuple[int, int]]) -> str:
    if r is None:
        return "any"
    a, b = r
    return f"{a}" if a == b else f"{a}-{b}"


# === Atomic helpers ===========================================================

def gold_buckets_for_item(required_paths: List[str],
                          optional_paths: List[str],
                          gold_lines_map: Dict[str, Optional[Tuple[int, int]]],
                          optional_map: Dict[str, bool]) -> Dict[str, Tuple[Optional[Tuple[int,int]], bool]]:
    """Return { path -> (gold_range_or_None, is_optional_bool) }"""
    buckets = {}
    for p in set(required_paths + optional_paths):
        buckets[p] = (gold_lines_map.get(p, None), bool(optional_map.get(p, False)))
    return buckets


def atomic_hit_status(path: str,
                      pred_line: Optional[int],
                      buckets: Dict[str, Tuple[Optional[Tuple[int,int]], bool]],
                      tol: int) -> Optional[str]:
    """
    Returns:
      'required' | 'optional' if (path,line) hits a gold bucket atomically
      None otherwise
    """
    if path not in buckets:
        return None
    gold_range, is_opt = buckets[path]
    if gold_range is None:  # 'any'
        return 'optional' if is_opt else 'required'
    if pred_line is None:
        return None
    if within_tolerance(pred_line, gold_range, tol):
        return 'optional' if is_opt else 'required'
    return None


# === HumanEval-style pass@k estimator =======================================

def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k) using numerically stable product form.
    """
    n, c = int(num_samples), int(num_correct)
    if n - c < k:
        return 1.0
    arr = np.arange(n - c + 1, n + 1, dtype=float)
    return float(1.0 - np.prod(1.0 - k / arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset JSON.")
    ap.add_argument("--code-context", required=True, help="Path to gptree_output.txt (or similar).")
    ap.add_argument("--model", required=True, help="OpenAI model name, e.g., gpt-4o, gpt-4o-mini, gpt-4.1-mini.")
    ap.add_argument("--outdir", default="runs", help="Directory to write CSV logs.")
    ap.add_argument("--line-tolerance", type=int, default=10, help="± line tolerance outside gold range.")
    ap.add_argument("--max-context-chars", type=int, default=120000, help="Truncate repo context to this many chars.")
    ap.add_argument("--stop-on-fp", action="store_true",
                    help="Interactively adjudicate UNION FPs (modifies dataset on 'yes').")
    ap.add_argument("--num-runs", type=int, default=1, help="Number of independent runs per example (k).")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset_obj = json.load(f)
    code_context = load_text(args.code_context, max_chars=args.max_context_chars)
    flat_items = flatten_dataset(dataset_obj)

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    csv_path = os.path.join(args.outdir, f"eval_{dataset_name}_{args.model}_{timestamp}.csv")

    rows: List[Dict[str, Any]] = []

    # Aggregates (macro across examples)
    agg_perrun_precision = 0.0
    agg_perrun_recall = 0.0
    agg_perrun_f1 = 0.0
    agg_perrun_em = 0.0

    agg_union_precision = 0.0
    agg_union_recall = 0.0
    agg_union_f1 = 0.0
    agg_union_em = 0.0

    # macro Pass@{1,3,5,k}
    agg_pass_at_1 = 0.0
    agg_pass_at_3 = 0.0
    agg_pass_at_5 = 0.0
    agg_pass_at_k = 0.0

    for idx, item in enumerate(flat_items[:10], 1): # DEBUG!!!!
        tag = item["tag"]
        superfeature = item["superfeature"]
        feature_desc = item["feature_desc"]
        required_paths: List[str] = item["required_paths"]
        optional_paths: List[str] = item["optional_paths"]
        gold_lines_map: Dict[str, Optional[Tuple[int, int]]] = item["gold_lines_map"]
        optional_map: Dict[str, bool] = item["optional_map"]

        # Prompt
        short_prompt = build_task_text(tag, superfeature, feature_desc)
        messages = prompt_template(tag, superfeature, feature_desc, code_context)

        # Runs
        run_raws: List[str] = []
        run_pred_sets: List[Set[str]] = []
        run_pred_lines: List[Dict[str, Optional[int]]] = []
        perrun_precisions: List[float] = []
        perrun_recalls: List[float] = []
        perrun_f1s: List[float] = []
        perrun_ems: List[int] = []  # strict EM per run (FN=0 and FP=0)

        set_required: Set[str] = set(required_paths)
        set_optional: Set[str] = set(optional_paths)
        set_gold_all: Set[str] = set_required | set_optional

        for r in range(max(1, args.num_runs)):
            try:
                raw = call_openai(client, args.model, messages)
            except Exception as e:
                print(f"[{idx}/{len(flat_items)}] OpenAI call failed (run {r+1}): {e}", file=sys.stderr)
                raw = ""
            run_raws.append(raw)

            preds = extract_json_list(raw)
            pred_paths: List[str] = []
            pred_lines_map: Dict[str, Optional[int]] = {}
            for obj in preds:
                fp = obj.get("file_path")
                ln = obj.get("line_number", None)
                if isinstance(ln, str) and ln.strip().lower() in {"null", "none", "any", ""}:
                    ln = None
                elif isinstance(ln, (int, float)):
                    ln = int(ln)
                else:
                    ln = None if ln is None else None
                if isinstance(fp, str) and fp.strip():
                    path_clean = fp.strip()
                    pred_paths.append(path_clean)
                    pred_lines_map[path_clean] = ln

            set_pred: Set[str] = set(pred_paths)
            run_pred_sets.append(set_pred)
            run_pred_lines.append(pred_lines_map)

            # Per-run atomic metrics
            buckets = gold_buckets_for_item(required_paths, optional_paths, gold_lines_map, optional_map)
            tp_req = tp_opt = fp_ = 0
            matched_req: Set[str] = set()

            for p in set_pred:
                status = atomic_hit_status(p, pred_lines_map.get(p, None), buckets, args.line_tolerance)
                if status == 'required':
                    tp_req += 1
                    matched_req.add(p)
                elif status == 'optional':
                    tp_opt += 1
                else:
                    fp_ += 1

            fn_req = len(set_required - matched_req)
            total_pred = len(set_pred)
            m = compute_metrics(tp_total=tp_req + tp_opt, fp=fp_, fn_required=fn_req,
                                total_required_gold=len(set_required), total_pred=total_pred,
                                tp_required=tp_req, tp_optional=tp_opt)
            perrun_precisions.append(m["precision"])
            perrun_recalls.append(m["recall"])
            perrun_f1s.append(m["f1"])
            # STRICT EM: must have all required and NO FPs
            perrun_ems.append(1 if (fn_req == 0 and fp_ == 0) else 0)

        # Print block (verbose for traceability)
        print(f"\n=== [{idx}/{len(flat_items)}] Evaluating: {tag} — {superfeature} ===")
        print("-" * 80)
        print("Prompt (no context)")
        print("-" * 80)
        print(short_prompt)
        print("-" * 80)
        print("Gold answers from dataset")
        print("-" * 80)
        all_gold = required_paths + optional_paths
        if all_gold:
            for p in sorted(set(all_gold)):
                label_opt = " [opt]" if optional_map.get(p) else ""
                print(f"{p} @ {fmt_line_range(gold_lines_map.get(p))}{label_opt}")
        else:
            print("(none)")
        print("-" * 80)
        print("Raw LLM response(s)")
        print("-" * 80)
        if run_raws:
            def _pp(s, max_chars=4000):
                s = (s or "").strip()
                return s if len(s) <= max_chars else s[:max_chars] + "\n...[TRUNCATED IN CONSOLE]..."
            for ri, raw in enumerate(run_raws, 1):
                print(f"[Run {ri}] {_pp(raw)}\n")
        else:
            print("(empty)")
        print("=" * 80 + "\n")

        # Per-run macro (this example)
        ex_perrun_precision = float(np.mean(perrun_precisions)) if perrun_precisions else 0.0
        ex_perrun_recall = float(np.mean(perrun_recalls)) if perrun_recalls else 0.0
        ex_perrun_f1 = float(np.mean(perrun_f1s)) if perrun_f1s else 0.0
        ex_perrun_em = float(np.mean(perrun_ems)) if perrun_ems else 0.0

        agg_perrun_precision += ex_perrun_precision
        agg_perrun_recall += ex_perrun_recall
        agg_perrun_f1 += ex_perrun_f1
        agg_perrun_em += ex_perrun_em

        # UNION metrics (atomic)
        union_pred_paths: Set[str] = set().union(*run_pred_sets) if run_pred_sets else set()

        # Adjudication before scoring
        adjudicated = False
        if args.stop_on_fp and union_pred_paths:
            set_required = set(required_paths); set_optional = set(optional_paths)
            set_gold_all = set_required | set_optional
            union_fps_paths = sorted(list(union_pred_paths - set_gold_all))
            if union_fps_paths:
                print("\n--- Human adjudication required (UNION) ---")
                print(f"Feature: [{tag}] {superfeature} -> {feature_desc}")
                for fp_candidate in list(union_fps_paths):
                    print(f"\nPredicted file not in gold (union): {fp_candidate}")
                    action = input(
                        "Press Enter to add as OPTIONAL(any). Type 'req' to add as REQUIRED. "
                        "Type 'skip' to leave as FP: "
                    ).strip().lower()

                    if action not in {"", "req"}:
                        print("Keeping as FP.")
                        continue

                    ln = input(
                        "Provide gold line range (e.g., '147-153', '147', or press Enter for 'any'): "
                    ).strip().lower()
                    lines_value = "any" if ln in {"", "any", "unknown", "null", "n/a"} else ln

                    try:
                        feat = dataset_obj[item["g_idx"]]["features"][item["f_idx"]]
                        if "file_paths" not in feat or not isinstance(feat["file_paths"], list):
                            feat["file_paths"] = []
                        feat["file_paths"].append({
                            "path": fp_candidate,
                            "lines": lines_value,
                            "optional": (action != "req"),
                            "code_preview": []
                        })
                        with open(args.dataset, "w", encoding="utf-8") as f:
                            json.dump(dataset_obj, f, ensure_ascii=False, indent=2)

                        is_opt = (action != "req")
                        optional_map[fp_candidate] = is_opt
                        gold_lines_map[fp_candidate] = parse_line_range(lines_value)
                        if is_opt:
                            set_optional.add(fp_candidate)
                        else:
                            set_required.add(fp_candidate)
                        set_gold_all.add(fp_candidate)

                        adjudicated = True
                        union_pred_paths.add(fp_candidate)

                        print(f"Added {fp_candidate} as {'OPTIONAL' if is_opt else 'REQUIRED'} "
                              f"with lines='{lines_value}'.")
                    except Exception as e:
                        print(f"Failed to update dataset with '{fp_candidate}': {e}")

        # Build path -> list of predicted lines across runs
        path_to_lines: Dict[str, List[Optional[int]]] = {p: [] for p in union_pred_paths}
        for lines_map in run_pred_lines:
            for p in union_pred_paths:
                path_to_lines[p].append(lines_map.get(p, None))

        buckets_union = gold_buckets_for_item(required_paths, optional_paths, gold_lines_map, optional_map)

        union_hit_required: Set[str] = set()
        union_hit_optional: Set[str] = set()
        union_fp = 0

        for p, cand_lines in path_to_lines.items():
            status = None
            if p in buckets_union:
                gold_range, is_opt = buckets_union[p]
                if gold_range is None:
                    status = 'optional' if is_opt else 'required'
                else:
                    for ln in cand_lines:
                        if ln is not None and within_tolerance(ln, gold_range, args.line_tolerance):
                            status = 'optional' if is_opt else 'required'
                            break
            if status == 'required':
                union_hit_required.add(p)
            elif status == 'optional':
                union_hit_optional.add(p)
            else:
                union_fp += 1

        set_required = set(required_paths)
        union_tp_required = len(union_hit_required)
        union_tp_optional = len(union_hit_optional)
        union_tp_total = union_tp_required + union_tp_optional
        union_fn_required = len(set_required - union_hit_required)

        union_pred_count = len(union_pred_paths)
        union_precision = union_tp_total / union_pred_count if union_pred_count > 0 else 0.0
        union_recall = union_tp_required / len(set_required) if len(set_required) > 0 else 0.0
        union_f1 = (2 * union_precision * union_recall / (union_precision + union_recall)
                    if union_precision + union_recall > 0 else 0.0)
        union_em = 1.0 if union_fn_required == 0 else 0.0

        agg_union_precision += union_precision
        agg_union_recall += union_recall
        agg_union_f1 += union_f1
        agg_union_em += union_em

        # Pass@{1,3,5,k} (strict EM per run)
        n = max(1, args.num_runs)
        c = int(sum(perrun_ems))
        ks = [1, 3, 5, n]
        pass_scores = {}
        for kk in ks:
            kk_eff = min(n, kk)
            pass_scores[f"pass_at_{kk}"] = estimate_pass_at_k(num_samples=n, num_correct=c, k=kk_eff)

        # Aggregate these
        agg_pass_at_1 += pass_scores["pass_at_1"]
        agg_pass_at_3 += pass_scores["pass_at_3"]
        agg_pass_at_5 += pass_scores["pass_at_5"]
        agg_pass_at_k += pass_scores[f"pass_at_{n}"]

        # CSV row
        gold_with_lines = ";".join(
            f"{p}@{fmt_line_range(gold_lines_map.get(p))}{('[opt]' if optional_map.get(p) else '')}"
            for p in sorted((set(required_paths) | set(optional_paths)))
        )
        row = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "model": args.model,
            "superfeature": superfeature,
            "tag": tag,
            "feature_desc": feature_desc,
            "num_runs": n,
            "gold_required_paths": to_csv_safe_list(sorted(set_required)),
            "gold_optional_paths": to_csv_safe_list(sorted(set(optional_paths))),
            "gold_with_lines": gold_with_lines,
            # Per-run macro (this example)
            "perrun_macro_precision": round(ex_perrun_precision, 4),
            "perrun_macro_recall_required": round(ex_perrun_recall, 4),
            "perrun_macro_f1": round(ex_perrun_f1, 4),
            "perrun_macro_em": round(ex_perrun_em, 4),
            # Union metrics (this example)
            "union_pred_paths": to_csv_safe_list(sorted(union_pred_paths)),
            "union_TP_total": union_tp_total,
            "union_TP_required": union_tp_required,
            "union_TP_optional": union_tp_optional,
            "union_FP": union_fp,
            "union_FN_required": union_fn_required,
            "union_precision": round(union_precision, 4),
            "union_recall_required": round(union_recall, 4),
            "union_f1": round(union_f1, 4),
            "union_em": round(union_em, 4),
            # Pass@{1,3,5,k}
            "pass_at_1": round(pass_scores["pass_at_1"], 4),
            "pass_at_3": round(pass_scores["pass_at_3"], 4),
            "pass_at_5": round(pass_scores["pass_at_5"], 4),
            "pass_at_k": round(pass_scores[f"pass_at_{n}"], 4),
            # Logs
            "adjudicated": adjudicated,
            "prompt_no_context": short_prompt,
            "raw_llm_response_all_runs": "\n\n---\n\n".join(run_raws),
            # Optional: store the per-run EM vector for transparency/debugging
            "perrun_em_vector": "".join(map(str, perrun_ems)),
        }
        rows.append(row)

        # Short console summary
        print(f"[{idx}/{len(flat_items)}] {tag} | {superfeature} | "
              f"PerRunMacro P:{row['perrun_macro_precision']} R:{row['perrun_macro_recall_required']} "
              f"F1:{row['perrun_macro_f1']} EM:{row['perrun_macro_em']} "
              f"| Union P:{row['union_precision']} R:{row['union_recall_required']} F1:{row['union_f1']} EM:{row['union_em']} "
              f"| Pass@1:{row['pass_at_1']} Pass@3:{row['pass_at_3']} Pass@5:{row['pass_at_5']} Pass@k:{row['pass_at_k']}")

    # Write CSV
    if rows:
        if HAS_PANDAS:
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        else:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    else:
        fieldnames = [
            "timestamp","dataset","model","superfeature","tag","feature_desc","num_runs",
            "gold_required_paths","gold_optional_paths","gold_with_lines",
            "perrun_macro_precision","perrun_macro_recall_required","perrun_macro_f1","perrun_macro_em",
            "union_pred_paths","union_TP_total","union_TP_required","union_TP_optional",
            "union_FP","union_FN_required","union_precision","union_recall_required","union_f1","union_em",
            "pass_at_1","pass_at_3","pass_at_5","pass_at_k","adjudicated","prompt_no_context",
            "raw_llm_response_all_runs","perrun_em_vector"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    # Macro across examples
    N = len(rows) if rows else 1
    print("\n=== Aggregate (macro-averaged across examples) ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_name}")
    print(f"Features evaluated: {len(flat_items)}")
    print(f"Per-run macro: Precision={agg_perrun_precision/N:.4f} "
          f"Recall(required)={agg_perrun_recall/N:.4f} "
          f"F1={agg_perrun_f1/N:.4f} "
          f"EM={agg_perrun_em/N:.4f}")
    print(f"Union: Precision={agg_union_precision/N:.4f} "
          f"Recall(required)={agg_union_recall/N:.4f} "
          f"F1={agg_union_f1/N:.4f} "
          f"EM={agg_union_em/N:.4f}")
    print(f"Pass@1={agg_pass_at_1/N:.4f}  Pass@3={agg_pass_at_3/N:.4f}  "
          f"Pass@5={agg_pass_at_5/N:.4f}  Pass@k={agg_pass_at_k/N:.4f}")
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
