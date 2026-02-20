# llm_pipeline.py

import json
import requests
from typing import Dict, List

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"


def run_local_llm(components: List[Dict]) -> Dict:
    prompt = build_prompt(components)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()
    text = result.get("response", "").strip()

    return parse_llm_response(text)


def build_prompt(components: List[Dict]) -> str:
    num = len(components)
    if num == 0:
        raise ValueError("No components detected")

    areas = [c["area"] for c in components]
    norm_areas = [c["normalized_area"] for c in components]

    type_counts = {}
    size_counts = {}

    for c in components:
        t = c.get("type", "unknown")
        s = c.get("size", "unknown")

        type_counts[t] = type_counts.get(t, 0) + 1
        size_counts[s] = size_counts.get(s, 0) + 1

    stats = {
        "count": num,
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / num,
        "coverage": sum(norm_areas),
        "type_counts": type_counts,
        "size_counts": size_counts
    }

    return f"""
You are analyzing a PCB based ONLY on structured component data.

Summary:
{json.dumps(stats, indent=2)}

First 10 components:
{json.dumps(components[:10], indent=2)}

Guidelines:
- Use type_counts to estimate composition
- IC count strongly affects cost
- Many small components → higher complexity
- Coverage indicates board density
- If data is uncertain, say so

Task:
Estimate:
1. Board complexity (low / medium / high)
2. Likely PCB type (power / MCU / sensor / mixed / unknown)
3. Approximate BOM cost in INR (give a range)
4. Short reasoning

Respond ONLY with valid JSON:

{{
  "complexity": "...",
  "pcb_type": "...",
  "estimated_bom_inr": "...",
  "reasoning": "..."
}}
"""


def parse_llm_response(text: str) -> Dict:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception:
        return {
            "raw_model_output": text
        }


def run_llm(messages: List[Dict], tools: List[Dict]) -> Dict:
    system_instructions = """
You are a STRICT JSON API for PCB analysis.

You must ONLY return valid JSON.
No explanations, no extra text.

You analyze PCB data using CV and OCR signals.

IMPORTANT: OCR INTERPRETATION RULES

OCR text may contain:
1. Reference designators:
   - Rxx → resistor
   - Cxx → capacitor
   - Uxx → IC
   - Jxx → connector

These are NOT component names.
They only indicate component category.

2. Part markings:
   - Alphanumeric codes like LM358, ATMEGA328
   These MAY be real IC names, but are uncertain.

Rules:
- Do NOT treat all OCR text as IC names.
- Use OCR only as supporting evidence.
- Use CV-derived component counts as primary signal.
- OCR is noisy and incomplete.

COST ESTIMATION RULES:
- Do NOT assume fixed prices.
- Use approximate ranges:
  - resistor: 0.5–5 INR
  - capacitor: 1–50 INR
  - IC: 20–500+ INR
- Return cost as a RANGE, not a single value.

Allowed outputs:

1) Tool call:
{
  "tool_call": {
    "name": "...",
    "arguments": { ... }
  }
}

2) Final answer:
{
  "final_answer": {
    "complexity": "...",
    "pcb_type": "...",
    "estimated_bom_inr": "min-max INR",
    "reasoning": "..."
  }
}
"""
    full_prompt = system_instructions + "\n\nConversation:\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        full_prompt += f"{role.upper()}: {content}\n"

    full_prompt += "\nAvailable Tools:\n"

    for tool in tools:
        full_prompt += f"""
Tool: {tool['name']}
Description: {tool['description']}
Parameters: {json.dumps(tool['parameters'], indent=2)}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()
    text = result.get("response", "").strip()

    return parse_agent_response(text)

def parse_agent_response(text: str) -> Dict:
    import json

    # try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # extract largest json block
    try:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)

    except:
        pass

    return {
        "error": "Invalid JSON format",
        "raw_output": text
    }