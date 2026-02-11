# tools.py

import json
from cv_pipeline import run_cv

def run_cv_tool(image_path: str):

    result = run_cv(image_path)

    if result.get("object_type") != "PCB":
        return {
            "object_type": result.get("object_type"),
            "component_count": 0
        }

    components = result.get("components", [])

    if not components:
        return {
            "object_type": "PCB",
            "component_count": 0
        }

    areas = [c["area"] for c in components]
    norm_areas = [c["normalized_area"] for c in components]

    summary = {
        "object_type": "PCB",
        "component_count": len(components),
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / len(areas),
        "coverage": sum(norm_areas),
        "ic_count": len(result.get("IC name", [])),
        "bucket_summary": result.get("Bucket components", {})
    }

    return summary



# Tool Registry
TOOLS = {
    "run_cv": {
        "description": "Detect PCB and extract summarized component features from image.",
        "function": run_cv_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to input image"
                }
            },
            "required": ["image_path"]
        }
    }
}


def execute_tool(tool_name: str, arguments: dict):
    if tool_name not in TOOLS:
        raise ValueError(f"Tool {tool_name} not found.")

    tool = TOOLS[tool_name]
    func = tool["function"]

    return func(**arguments)


def get_tool_specs():
    specs = []

    for name, tool in TOOLS.items():
        specs.append({
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"]
        })

    return specs
