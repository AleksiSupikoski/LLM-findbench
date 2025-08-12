import json
import csv
from config import (
    DATASET_FILE, CONTEXT_FILE, OUTPUT_CSV_FILE, CSV_FIELDS, MODELS_TO_TEST
)
from utils import read_context, normalize_path
from llm import query_llm
from evaluation import evaluate_category


def main():
    with open(DATASET_FILE, encoding="utf-8") as f:
        data = json.load(f)
    context = read_context(CONTEXT_FILE)

    with open(OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for model in MODELS_TO_TEST:
            for item in data[:1]:  # --------------------------------- debug, do only first data object
                feature_name = item["feature"]
                feature_desc = item["nl_desc"]

                prompt = (
                    f"Given the super‑feature “{feature_name}: {feature_desc}”, locate all its sub‑features in each category:\n"
                    "- ui_layer, presentation_logic, business_logic,\n"
                    "  data_fetch_persistence, state_management,\n"
                    "  event_handling, validation, dependency_layer.\n"
                    "Return ONLY a JSON object whose keys are these categories,\n"
                    "each mapping to an array (or null) of objects with:\n"
                    "  \"file_path\": a path to the file in codebase where the sub-feature is implemented (string), \"func_class_object\": a function, class, object definition or variable and such if applicable (string|null), \"lines\": lines range (e.g. \"10-25\" can be multiple ranges).\n"
                    f"Context:\n{context}\n"
                )

                pred, resp_time, raw = query_llm(prompt, model, item if model == "mock" else None)

                print(f"\n\n ------------ [DEBUG] Model: {model} | Feature: {feature_name} | Response Time: {resp_time:.2f}s")
                print("---------------- RAW -------------------")
                print(raw)
                print("----------------------------------------")

                for cat in [
                    "ui_layer", "presentation_logic", "business_logic",
                    "data_fetch_persistence", "state_management",
                    "event_handling", "validation", "dependency_layer"
                ]:
                    gold_list = item.get(cat, {}).get("subfeatures", []) or []
                    pred_list = pred.get(cat) or []

                    precision, recall, f1, avg_iou, num_gold, num_pred, TP, FP, FN = evaluate_category(pred_list, gold_list)

                    print(f"\n[DEBUG] Category: {cat}")
                    print(f"Gold subfeatures: {len(gold_list)}")
                    for g in gold_list:
                        print(f"  - {normalize_path(g['file_path'])} {g['lines']}")
                    print(f"Predicted subfeatures: {len(pred_list)}")
                    for p in pred_list:
                        print(f"  - {normalize_path(p['file_path'])} {p['lines']}")
                    print(f"TP: {TP}, FP: {FP}, FN: {FN}, Avg IoU: {avg_iou:.3f}")
                    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

                    writer.writerow({
                        "model": model,
                        "feature": feature_desc,
                        "category": cat,
                        "precision": f"{precision:.3f}",
                        "recall": f"{recall:.3f}",
                        "f1": f"{f1:.3f}",
                        "avg_iou": f"{avg_iou:.3f}",
                        "num_gold": num_gold,
                        "num_pred": num_pred,
                        "raw_model_output": raw.replace("\n", " ")
                    })

    print(f"\nEvaluation complete. Results saved to: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()