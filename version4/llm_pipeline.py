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
    
def run_llm(messages: List[Dict], tools: List[Dict]) -> Dict:
    """
    General LLM call that supports tool reasoning.
    messages: conversation history
    tools: list of tool specs
    """

    system_instructions = """
You are an autonomous PCB analysis agent.

Your goal:
Analyze an image of a printed circuit board (PCB) and estimate BOM characteristics using available tools.

You must reason step-by-step.

You have access to tools. Use them when needed.

Rules:
1. Do NOT guess information if it has not been observed.
2. If image analysis is required, call the appropriate tool.
3. If component data is available, analyze it before estimating BOM.
4. Only call one tool at a time.
5. When you are confident in your result, return a final_answer.
6. If insufficient data exists, explicitly state uncertainty.

STRICT TOOL RULES:
7. You can ONLY use tools listed in "Available Tools".
8. NEVER invent new tool names.
9. If the required information is already available, DO NOT call any tool.
10. Cost estimation must be done in final_answer, NOT via a tool.

Tool usage format:
If you need to call a tool, respond ONLY with:

{
  "tool_call": {
    "name": "tool_name",
    "arguments": { ... }
  }
}

If you are ready to conclude, respond ONLY with:

{
  "final_answer": {
    "complexity": "...",
    "pcb_type": "...",
    "estimated_bom_inr": "...",
    "reasoning": "..."
  }
}

Do not include explanations outside JSON.
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
    import re
    import json

    matches = re.findall(r'\{.*\}', text, re.DOTALL)

    if not matches:
        return {"error": "Invalid LLM response", "raw_output": text}

    # try parsing from last json block
    for block in reversed(matches):
        try:
            return json.loads(block)
        except:
            continue

    return {"error": "Invalid JSON format", "raw_output": text}

