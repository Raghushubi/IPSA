# agent_tools.py

import cv2
from cv_pipeline import run_cv
from ocr import read_full_image_text, extract_reference_counts, filter_ic_candidates

# cache for cv results
CV_CACHE = {}


def safe_error(msg):
    return {
        "error": msg
    }


def get_cv_result(image_path: str):
    try:
        if image_path not in CV_CACHE:
            CV_CACHE[image_path] = run_cv(image_path)
        return CV_CACHE[image_path] or {}
    except Exception as e:
        return safe_error(f"cv_failed: {str(e)}")


# tool implementations
def run_cv_tool(image_path: str):
    try:
        result = get_cv_result(image_path)

        if "error" in result:
            return result

        if result.get("object_type") != "PCB":
            return {
                "object_type": result.get("object_type"),
                "component_count": 0
            }

        components = result.get("components", []) or []

        if not components:
            return {
                "object_type": "PCB",
                "component_count": 0
            }

        areas = [c.get("area", 0) for c in components if isinstance(c, dict)]
        norm_areas = [c.get("normalized_area", 0) for c in components if isinstance(c, dict)]

        if not areas:
            return {
                "object_type": "PCB",
                "component_count": 0
            }

        type_counts = {}
        size_counts = {}

        for c in components:
            if not isinstance(c, dict):
                continue

            t = c.get("type", "unknown")
            s = c.get("size", "unknown")

            type_counts[t] = type_counts.get(t, 0) + 1
            size_counts[s] = size_counts.get(s, 0) + 1

        summary = {
            "object_type": "PCB",
            "component_count": len(components),
            "min_area": min(areas),
            "max_area": max(areas),
            "mean_area": sum(areas) / len(areas) if areas else 0,
            "coverage": sum(norm_areas) if norm_areas else 0,
            "type_counts": type_counts,
            "size_counts": size_counts
        }

        return summary

    except Exception as e:
        return safe_error(f"run_cv_tool_failed: {str(e)}")


def get_component_stats_tool(image_path: str):
    try:
        result = get_cv_result(image_path)

        if "error" in result:
            return result

        if result.get("object_type") != "PCB":
            return {
                "component_count": 0,
                "coverage": 0.0
            }

        components = result.get("components", []) or []

        if not components:
            return {
                "component_count": 0,
                "coverage": 0.0
            }

        areas = [c.get("area", 0) for c in components if isinstance(c, dict)]
        norm_areas = [c.get("normalized_area", 0) for c in components if isinstance(c, dict)]

        if not areas:
            return {
                "component_count": 0,
                "coverage": 0.0
            }

        type_counts = {}
        for c in components:
            if not isinstance(c, dict):
                continue

            t = c.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "component_count": len(components),
            "min_area": min(areas),
            "max_area": max(areas),
            "mean_area": sum(areas) / len(areas) if areas else 0,
            "coverage": sum(norm_areas) if norm_areas else 0,
            "type_counts": type_counts
        }

    except Exception as e:
        return safe_error(f"component_stats_failed: {str(e)}")


def get_ic_info_tool(image_path: str):
    try:
        image = cv2.imread(image_path)

        if image is None:
            return safe_error("image_load_failed")

        texts = read_full_image_text(image) or []
        ref_counts = extract_reference_counts(texts) or {"R": 0, "C": 0, "U": 0, "J": 0}
        ic_names = filter_ic_candidates(texts) or []

        # debug block
        try:
            print("\n--- OCR RAW TEXT (first 20) ---")
            print(texts[:20])

            print("\n--- OCR REFERENCE COUNTS ---")
            print(ref_counts)

            print("\n--- OCR IC CANDIDATES ---")
            print(ic_names[:10])
        except Exception:
            pass

        return {
            "reference_counts": ref_counts,
            "ic_count_ocr": len(ic_names),
            "possible_ic_names": ic_names[:10]
        }

    except Exception as e:
        return safe_error(f"ocr_failed: {str(e)}")


# tool execution
def execute_tool(tool_name: str, arguments: dict):
    try:
        if tool_name not in TOOLS:
            return safe_error(f"tool_not_found: {tool_name}")

        tool = TOOLS[tool_name]
        func = tool.get("function")

        if func is None:
            return safe_error(f"invalid_tool_function: {tool_name}")

        arguments = arguments or {}

        return func(**arguments)

    except Exception as e:
        return safe_error(f"tool_execution_failed: {str(e)}")


def get_tool_specs():
    specs = []

    for name, tool in TOOLS.items():
        specs.append({
            "name": name,
            "description": tool.get("description"),
            "parameters": tool.get("parameters")
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