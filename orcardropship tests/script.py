import json
import os
import time
import csv
from typing import List, Dict, Any, Tuple, Optional

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
}

MODELS_TO_TEST = ["mock"]  # or like "openai-gpt-4", "gemini-1", etc.

IOU_SUCCESS_THRESHOLD = 0.5
OUTPUT_CSV_FILE = "evaluation_results.csv"
DATASET_FILE = "dataset.json"
CONTEXT_FILE = "gptree.txt"


def read_context() -> str:
    try:
        with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Context file '{CONTEXT_FILE}' not found, proceeding without extra context.")
        return ""


first_mock_call = True
def query_mock_model(prompt: str, dataset_item: Dict[str, Any]) -> Dict[str, Any]:
    global first_mock_call
    time.sleep(0.5)

    if first_mock_call:
        first_mock_call = False
        return {
            "file_path": dataset_item["file-path"],
            "lines": "10-25"
        }

    # mock data with perfect answers
    return {
        "file_path": dataset_item["file-path"],
        "lines": dataset_item["lines"]
    }


def query_llm_api(prompt: str, model_name: str, dataset_item: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], float, str]:
    start_time = time.time()
    raw_response_str = ""

    try:
        if model_name == "mock":
            response_obj = query_mock_model(prompt, dataset_item)
            raw_response_str = json.dumps(response_obj)
            parsed_json = response_obj

        elif model_name.startswith("openai-"):
            import openai
            openai.api_key = API_KEYS["openai"]
            openai_model = model_name[len("openai-"):]
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw_response_str = response.choices[0].message.content
            parsed_json = json.loads(raw_response_str)

        elif model_name.startswith("gemini-"):
            # TODO: Add Gemini API call
            return None, 0.0, "Gemini not implemented"

            # etc for other models

        else:
            raise NotImplementedError(f"API call for model '{model_name}' not implemented.")

    except Exception as e:
        print(f"Error querying {model_name} or parsing response: {e}")
        parsed_json = None
        raw_response_str = f"ERROR: {e}"

    response_time = time.time() - start_time
    return parsed_json, response_time, raw_response_str


def parse_lines(lines_str: str) -> Optional[Tuple[int, int]]:
    try:
        parts = lines_str.strip().split('-')
        if len(parts) != 2:
            return None
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, AttributeError):
        return None


def calculate_iou(lines_true_str: str, lines_pred_str: str) -> float:
    true_range = parse_lines(lines_true_str)
    pred_range = parse_lines(lines_pred_str)

    if not true_range or not pred_range:
        return 0.0

    true_set = set(range(true_range[0], true_range[1] + 1))
    pred_set = set(range(pred_range[0], pred_range[1] + 1))

    intersection = len(true_set.intersection(pred_set))
    union = len(true_set.union(pred_set))

    return intersection / union if union > 0 else 0.0


def main():
    print("LLM evaluation script...")

    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} items from {DATASET_FILE}")
    except FileNotFoundError:
        print(f"Dataset file '{DATASET_FILE}' not found. Exiting.")
        return
    except json.JSONDecodeError:
        print(f"Could not parse JSON in '{DATASET_FILE}'. Check file format.")
        return

    context = read_context()

    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'model', 'nl_desc', 'true_file_path', 'true_lines',
            'predicted_file_path', 'predicted_lines', 'response_time',
            'path_is_correct', 'lines_iou', 'is_mostly_correct', 'raw_model_output'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in MODELS_TO_TEST:
            print(f"\n--- Testing Model: {model_name} ---")
            for i, item in enumerate(dataset):
                nl_desc = item.get("nl-desc")
                true_path = item.get("file-path")
                true_lines = item.get("lines")

                if not all([nl_desc, true_path, true_lines]):
                    print(f"Skipping invalid dataset item: {item}")
                    continue

                print(f"  Test {i + 1}/{len(dataset)}: '{nl_desc[:50]}...'")

                prompt = (
                    f"Find and locate file path and line numbers range where this feature is implemented: '{nl_desc}'. "
                    "Respond ONLY with a single JSON object containing two keys: 'file_path' (string) and 'lines' (string, e.g., '10-25'). "
                    f"Context: {context}\n\n"
                )

                prediction, resp_time, raw_output = query_llm_api(
                    prompt, model_name, dataset_item=item if model_name == "mock" else None
                )

                pred_path = prediction.get('file_path') if prediction else "N/A"
                pred_lines = prediction.get('lines') if prediction else "N/A"

                path_correct = (pred_path == true_path)

                if not path_correct:
                    iou_score = 0.0
                    mostly_correct = False
                else:
                    iou_score = calculate_iou(true_lines, pred_lines)
                    mostly_correct = iou_score >= IOU_SUCCESS_THRESHOLD

                writer.writerow({
                    'model': model_name,
                    'nl_desc': nl_desc,
                    'true_file_path': true_path,
                    'true_lines': true_lines,
                    'predicted_file_path': pred_path,
                    'predicted_lines': pred_lines,
                    'response_time': f"{resp_time:.4f}",
                    'path_is_correct': path_correct,
                    'lines_iou': f"{iou_score:.4f}",
                    'is_mostly_correct': mostly_correct,
                    'raw_model_output': raw_output
                })

    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    main()
