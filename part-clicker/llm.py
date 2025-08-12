import json
import time
from typing import Any, Dict, Tuple 
from config import API_KEYS
from utils import normalize_path


with open("mock_answers.json", encoding="utf-8") as f:
    MOCK_ANSWERS = json.load(f)


def query_mock_model(prompt: str, item: Dict[str, Any]) -> Dict[str, Any]:
    feature_name = item["feature"]
    for ans in MOCK_ANSWERS:
        if ans["feature"] == feature_name:
            return {cat: ans.get(cat, []) for cat in [
                "ui_layer", "presentation_logic", "business_logic",
                "data_fetch_persistence", "state_management",
                "event_handling", "validation", "dependency_layer"
            ]}
    return {cat: [] for cat in [
        "ui_layer", "presentation_logic", "business_logic",
        "data_fetch_persistence", "state_management",
        "event_handling", "validation", "dependency_layer"
    ]}


def query_llm(prompt: str, model: str, item=None) -> Tuple[Dict[str, Any], float, str]: 
    start = time.time()
    raw = ""
    resp = {}

    if model == "mock":
        resp = query_mock_model(prompt, item)
        raw = json.dumps(resp)

    elif model.startswith("openai-"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=API_KEYS["openai"])
            m = model.split("-", 1)[1]

            chat = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = chat.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            raw = ""
            resp = {}
            return resp, time.time() - start, raw

    elif model.startswith("aalto-"):
        try:
            from langchain_openai import ChatOpenAI

            aalto_key = API_KEYS["aalto"]
            if not aalto_key:
                raise RuntimeError("AALTO_API_KEY is not set in .env")
            llm = ChatOpenAI(
                openai_api_base="https://ai-gateway.k8s.aalto.fi/v1",
                openai_api_key=aalto_key,
                model_name=model.split("-", 1)[1],
                temperature=0.0,
            )
            response = llm.invoke(prompt)
            raw = response.content
            print(raw)

            resp = json.loads(raw)

        except Exception as e:
            print(f"Aalto LLM call failed: {e}")
            resp = {cat: [] for cat in [
                "ui_layer", "presentation_logic", "business_logic",
                "data_fetch_persistence", "state_management",
                "event_handling", "validation", "dependency_layer"
            ]}

    else:
        raise NotImplementedError(model)

    try:
        resp = json.loads(raw)
    except Exception:
        resp = {}

    return resp, time.time() - start, raw