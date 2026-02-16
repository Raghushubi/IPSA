# agent.py

from tools import execute_tool, get_tool_specs
from llm_pipeline import run_llm
import json

def build_reasoning_input(state):
    data = {}

    if state.get("stats"):
        data.update({
            "component_count": state["stats"].get("component_count"),
            "mean_area": state["stats"].get("mean_area"),
            "coverage": state["stats"].get("coverage")
        })

    if state.get("ic_info"):
        data["ic_count"] = state["ic_info"].get("ic_count")

    return data

def run_agent(image_path: str, max_steps: int = 6):
    
    tools = get_tool_specs()

    messages = []
    state = {
        "image_path": image_path,
        "stats": None,
        "ic_info": None
    }

    used_tools = []  # track tool usage

    # initial user instruction
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

        # inject compact state
        messages_with_state = messages + [{
            "role": "system",
            "content": f"Current state:\n{json.dumps(state)}"
        }]

        print("\n--- STATE SIZE ---")
        print(len(json.dumps(state)))

        response = run_llm(messages_with_state, tools)

        print("\n--- LLM Response ---")
        print(response)

        # invalid response
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

            # tool existence check
            tool_names = [t["name"] for t in tools]
            if tool_name not in tool_names:
                return {
                    "status": "error",
                    "reason": f"unknown_tool: {tool_name}"
                }

            # prevent same tool twice in a row
            if used_tools and used_tools[-1] == tool_name:
                return {
                    "status": "error",
                    "reason": f"repeated_tool_call: {tool_name}"
                }

            # basic argument sanity
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

            # append to memory
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

            # build structured reasoning input
            reasoning_data = build_reasoning_input(state)

            # reflection prompt
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

            print("\n Reflection Response")
            print(reflection_response)

            if "final_answer" in reflection_response:
                final = reflection_response["final_answer"]

            return {
                "status": "success",
                "result": final,
                "steps_used": step + 1
            }


        # unknown output
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
