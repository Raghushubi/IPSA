# tools.py

import json
from cv_pipeline import run_cv

# cache for cv results
CV_CACHE = {}

def get_cv_result(image_path: str):
    if image_path not in CV_CACHE:
        CV_CACHE[image_path] = run_cv(image_path)
    return CV_CACHE[image_path]

# tool implementations
def run_cv_tool(image_path: str):
    """
    Full summary from CV pipeline
    """

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


def get_component_stats_tool(image_path: str):
    """
    Lightweight stats tool (no raw components exposed)
    """

    result = get_cv_result(image_path)

    if result.get("object_type") != "PCB":
        return {
            "component_count": 0,
            "min_area": 0,
            "max_area": 0,
            "mean_area": 0.0,
            "coverage": 0.0
        }

    components = result.get("components", [])

    if not components:
        return {
            "component_count": 0,
            "min_area": 0,
            "max_area": 0,
            "mean_area": 0.0,
            "coverage": 0.0
        }

    areas = [c["area"] for c in components]
    norm_areas = [c["normalized_area"] for c in components]

    return {
        "component_count": len(components),
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / len(areas),
        "coverage": sum(norm_areas)
    }


def get_ic_info_tool(image_path: str):
    """
    Extract IC count + names (lightweight)
    """

    result = get_cv_result(image_path)

    ic_names = result.get("IC name", [])

    return {
        "ic_count": len(ic_names),
        "ic_names": ic_names[:10]  # limit to avoid token blowup
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
