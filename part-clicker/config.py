import os
from dotenv import load_dotenv

load_dotenv()

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "aalto": os.getenv("AALTO_API_KEY", ""),
}

MODELS_TO_TEST = ["openai-gpt-5-nano"]  #-------------------------

IOU_SUCCESS_THRESHOLD = 0.5

DATASET_FILE = "dataset-fixed-struc-corrected-v4.json"
CONTEXT_FILE = "gptree_output.txt"
OUTPUT_CSV_FILE = "TEST-OPENAI-2nd-Eevaluation_results.csv"

CSV_FIELDS = [
    "model", "feature", "category",
    "precision", "recall", "f1", "avg_iou",
    "num_gold", "num_pred",
    "raw_model_output"
]
