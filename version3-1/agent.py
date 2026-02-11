# agent.py

from tools import execute_tool, get_tool_specs
from llm_pipeline import run_llm
import json

def run_agent(image_path: str, max_steps: int = 6):
    
    tools = get_tool_specs()

    messages = []

    # initial user instruction
    messages.append({
        "role": "user",
        "content": f"""
        New task:
        Analyze the PCB image at path: {image_path}

        Determine:
        - Board complexity
        - Likely PCB type
        - Estimated BOM cost in INR

        Use tools as needed.
        """
        })


    for step in range(max_steps):

        response = run_llm(messages, tools)

        print("\n--- LLM Response ---")
        print(response)

        # tool call
        if "tool_call" in response:
            tool_name = response["tool_call"]["name"]
            arguments = response["tool_call"]["arguments"]

            tool_result = execute_tool(tool_name, arguments)

            # append tool call and observation to memory
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
            return {
                "status": "success",
                "result": response["final_answer"],
                "steps_used": step + 1
            }

        else:
            return {
                "status": "error",
                "raw_response": response
            }

    return {
        "status": "max_steps_exceeded",
        "messages": messages
    }
