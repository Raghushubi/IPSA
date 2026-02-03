# llm_pipeline.py

import json
import requests
from typing import Dict, List

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"

def run_local_llm(components: List[Dict]) -> Dict:
    """
    Run local LLM on component geometry data.
    """
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
    """
    Build LLM prompt from component geometry.
    """
    num = len(components)
    if num == 0:
        raise ValueError("No components detected")
    
    areas = [c["area"] for c in components]
    norm_areas = [c["normalized_area"] for c in components]
    stats = {
        "count": num,
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / num,
        "coverage": sum(norm_areas)
    }
    
    return f"""
You are analyzing a PCB based ONLY on geometric component data.
Summary:
{json.dumps(stats, indent=2)}
First 10 components:
{json.dumps(components[:10], indent=2)}
Task:
Estimate:
1. Board complexity (low / medium / high)
2. Likely PCB type (power / MCU / sensor / mixed / unknown)
3. Approximate BOM cost in INR (Indian market)
4. Short reasoning
If unsure, say so explicitly.
Respond ONLY with valid JSON:
{{
  "complexity": "...",
  "pcb_type": "...",
  "estimated_bom_inr": "...",
  "reasoning": "..."
}}
"""


def parse_llm_response(text: str) -> Dict:
    """
    Extract JSON from LLM response text.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception:
        return {
            "raw_model_output": text
        }