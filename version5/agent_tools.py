# tools.py

import json
import cv2
from cv_pipeline import run_cv
from ocr import read_full_image_text, extract_reference_counts, filter_ic_candidates

# cache for cv results
CV_CACHE = {}

def get_cv_result(image_path: str):
    if image_path not in CV_CACHE:
        CV_CACHE[image_path] = run_cv(image_path)
    return CV_CACHE[image_path]

# tool implementations
def run_cv_tool(image_path: str):
    result = get_cv_result(image_path)

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

    type_counts = {}
    size_counts = {}

    for c in components:
        t = c.get("type", "unknown")
        s = c.get("size", "unknown")

        type_counts[t] = type_counts.get(t, 0) + 1
        size_counts[s] = size_counts.get(s, 0) + 1

    summary = {
        "object_type": "PCB",
        "component_count": len(components),
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / len(areas),
        "coverage": sum(norm_areas),
        "type_counts": type_counts,
        "size_counts": size_counts
    }

    return summary

def get_component_stats_tool(image_path: str):
    result = get_cv_result(image_path)

    if result.get("object_type") != "PCB":
        return {
            "component_count": 0,
            "coverage": 0.0
        }

    components = result.get("components", [])

    if not components:
        return {
            "component_count": 0,
            "coverage": 0.0
        }

    areas = [c["area"] for c in components]
    norm_areas = [c["normalized_area"] for c in components]

    type_counts = {}
    for c in components:
        t = c.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "component_count": len(components),
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / len(areas),
        "coverage": sum(norm_areas),
        "type_counts": type_counts
    }

def get_ic_info_tool(image_path: str):
    image = cv2.imread(image_path)

    try:
        texts = read_full_image_text(image)
        ref_counts = extract_reference_counts(texts)
        ic_names = filter_ic_candidates(texts)

    except Exception as e:
        print("\n--- OCR ERROR ---")
        print(e)
        raise e

    # DEBUG BLOCK
    print("\n--- OCR RAW TEXT (first 20) ---")
    print(texts[:20])

    print("\n--- OCR REFERENCE COUNTS ---")
    print(ref_counts)

    print("\n--- OCR IC CANDIDATES ---")
    print(ic_names[:10])

    return {
        "reference_counts": ref_counts,
        "ic_count_ocr": len(ic_names),
        "possible_ic_names": ic_names[:10]
    }

# tool execuion
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

# tool registry
TOOLS = {
    "run_cv": {
        "description": "Detect PCB and return summarized component features.",
        "function": run_cv_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string"
                }
            },
            "required": ["image_path"]
        }
    },

    "get_component_stats": {
        "description": "Get component statistics (count, area, coverage) from PCB image.",
        "function": get_component_stats_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string"
                }
            },
            "required": ["image_path"]
        }
    },

    "get_ic_info": {
        "description": "Get IC chip count and names from PCB image.",
        "function": get_ic_info_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string"
                }
            },
            "required": ["image_path"]
        }
    }
}
