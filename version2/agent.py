# agent.py

from typing import Dict
from cv_pipeline import run_cv
from llm_pipeline import run_local_llm


def run_agent(image_path: str) -> Dict:
    cv_result = run_cv(image_path)
    
    object_type = cv_result["object_type"]
    
    if object_type != "PCB":
        return {
            "cv": cv_result,
            "llm": None,
            "status": "non_pcb_object"
        }
    
    components = cv_result["components"]
    
    if len(components) == 0:
        return {
            "cv": cv_result,
            "llm": None,
            "status": "no_components"
        }
    
    llm_result = run_local_llm(components)
    
    return {
        "cv": cv_result,
        "llm": llm_result,
        "status": "success"
    }