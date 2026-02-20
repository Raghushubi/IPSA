# agent.py

from agent_tools import execute_tool, get_tool_specs
from llm_pipeline import run_llm
import json

def build_reasoning_input(state):
    data = {}

    # stats
    stats = state.get("stats")
    if stats:
        data.update({
            "component_count": stats.get("component_count"),
            "mean_area": stats.get("mean_area"),
            "coverage": stats.get("coverage"),
            "type_counts": stats.get("type_counts")
        })

    # ic info
    ic_info = state.get("ic_info")
    if ic_info:
        ic_count_cv = ic_info.get("ic_count_cv", 0)
        ic_count_ocr = ic_info.get("ic_count_ocr", 0)
        ic_names = ic_info.get("ic_names", [])

        data.update({
            "ic_count_cv": ic_count_cv,
            "ic_count_ocr": ic_count_ocr,
            "ic_names": ic_names
        })

        # derived signal
        data["ic_present"] = (ic_count_cv > 0) or (ic_count_ocr > 0)

    return data


def compact_state(state):
    stats = state.get("stats") or {}
    ic_info = state.get("ic_info") or {}

    return {
        "component_count": stats.get("component_count"),
        "mean_area": stats.get("mean_area"),
        "coverage": stats.get("coverage"),
        "type_counts": stats.get("type_counts"),
        "ic_count": ic_info.get("ic_count_ocr")
    }

def run_agent(image_path: str, max_steps: int = 6):
    
    tools = get_tool_specs()

    messages = []
    state = {
        "image_path": image_path,
        "stats": None,
        "ic_info": None
    }

    used_tools = []

    # initial instruction
    messages.append({
        "role": "user",
        "content": f"""
New task:
Analyze the PCB image at path: {image_path}

You must:
- Use tools to gather data
- Use ONLY observed data (do not guess)

Determine:
- Board complexity
- Likely PCB type
- Estimated BOM cost in INR

If data is insufficient, explicitly say so.
"""
    })


    for step in range(max_steps):

        # inject state
        messages_with_state = messages + [{
            "role": "system",
            "content": f"Current state:\n{json.dumps(compact_state(state))}"
        }]

        print("\n--- STEP ---", step + 1)
        print("\n--- STATE SIZE ---")
        print(len(json.dumps(state)))

        response = run_llm(messages_with_state, tools)

        print("\n--- LLM Response ---")
        print(response)

        if state["stats"] is not None and state["ic_info"] is None:
            # only override if llm is trying to finish early
            if "final_answer" in response:
                print("\n--- OVERRIDING: Forcing IC Info Tool ---")
                response = {
                    "tool_call": {
                        "name": "get_ic_info",
                        "arguments": {"image_path": image_path}
                    }
                }


        if not isinstance(response, dict):
            return {
                "status": "error",
                "reason": "invalid_response_format",
                "raw": response
            }

        # tool call 
        if "tool_call" in response:

            tool_name = response["tool_call"].get("name")
            arguments = response["tool_call"].get("arguments", {})

            tool_names = [t["name"] for t in tools]

            if tool_name not in tool_names:
                return {
                    "status": "error",
                    "reason": f"unknown_tool: {tool_name}"
                }

            if used_tools and used_tools[-1] == tool_name:
                return {
                    "status": "error",
                    "reason": f"repeated_tool_call: {tool_name}"
                }

            if arguments is None:
                arguments = {}

            try:
                tool_result = execute_tool(tool_name, arguments)
            except Exception as e:
                return {
                    "status": "error",
                    "reason": f"tool_execution_failed: {str(e)}"
                }

            used_tools.append(tool_name)

            # update state
            if tool_name == "get_component_stats":
                state["stats"] = tool_result

            elif tool_name == "get_ic_info":
                state["ic_info"] = tool_result

            elif tool_name == "run_cv":
                state["stats"] = tool_result

            # memory update
            messages.append({
                "role": "assistant",
                "content": json.dumps(response)
            })

            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result)
            })

            print("\n--- Tool Executed ---")
            print(tool_name)
            print(tool_result)

        # final answer
        elif "final_answer" in response:

            final = response["final_answer"]

            reasoning_data = build_reasoning_input(state)

            reflection_messages = [
                {
                    "role": "system",
                    "content": "You are reviewing your previous answer for correctness."
                },
                {
                    "role": "user",
                    "content": f"""
Original Answer:
{json.dumps(final)}

Observed Data:
{json.dumps(reasoning_data)}

Task:
Check if the answer is consistent with the data.

If correct, return same answer.
If incorrect or uncertain, revise it.

Respond ONLY in JSON format:
{{
  "final_answer": {{
    "complexity": "...",
    "pcb_type": "...",
    "estimated_bom_inr": "...",
    "reasoning": "..."
  }}
}}
"""
                }
            ]

            reflection_response = run_llm(reflection_messages, tools)

            print("\n--- Reflection Response ---")
            print(reflection_response)

            if isinstance(reflection_response, dict) and "final_answer" in reflection_response:
                final = reflection_response["final_answer"]

            return {
                "status": "success",
                "result": final,
                "steps_used": step + 1
            }

        # unknown
        else:
            return {
                "status": "error",
                "reason": "unknown_response_structure",
                "raw_response": response
            }

    # max steps
    return {
        "status": "max_steps_exceeded",
        "messages": messages
    }
