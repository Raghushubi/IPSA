In this version, the main change is in how the pipeline is structured. Earlier, the system followed a fixed flow: image → cv pipeline → llm → output. This worked, but everything was hardcoded, and the model had no control over what information to use.

In this version, I converted the pipeline into an agent-based flow. Instead of directly passing cv output to the llm, the model now runs in a loop. It can call tools, observe their outputs, update its internal state, and then decide what to do next. Finally, it returns an answer once it has enough information. So instead of a single pass, it now behaves like a step-by-step reasoning process.

cv pipeline and tools

The core cv pipeline is unchanged. All existing logic for pcb detection, component extraction, filtering, feature extraction, and basic ocr is kept the same.

Instead of modifying it, I created tool wrappers around it. This allows the agent to access specific information when needed.

Current tools:

run_cv → returns full summarized output

get_component_stats → returns numeric features like count, area, coverage

get_ic_info → returns ic count and detected text

This makes the system more modular, because the model does not need to use everything at once.

agent loop

The main change is the agent loop. Instead of a single call, the model now works iteratively:

decide which tool to call

execute tool

observe result

continue reasoning

return final answer

This allows more flexible reasoning compared to the earlier fixed pipeline.

state (memory)

A structured state dictionary is added inside the agent. This stores information such as stats and ic details.

After each tool call, the result is stored in the state. The state is then passed back to the model in the next step. This way, the model remembers what it has already observed and does not need to recompute everything.

control logic

Basic control logic is added to make the system stable.

This includes:

limiting the number of steps

handling invalid model responses

stopping if an unknown tool is called

Without this, the model sometimes tries to call tools that are not defined.

data simplification

Earlier, the full list of components was being passed to the model. This was large and noisy.

Now only summarized data is used:

component count

area statistics

coverage

ic count

This makes the output more stable and reduces unnecessary data.

reflection step

After generating the final answer, the model is asked once to review its own output.

If needed, it can correct the answer. In most cases, the answer remains the same, but this acts as a basic consistency check.

cv result reuse

Earlier, the cv pipeline was being executed multiple times for different tools.

Now a simple cache is added so that cv runs only once per image. The result is stored and reused by all tools. This reduces computation and keeps outputs consistent.

current output

The system now works end-to-end. The agent calls tools step by step and then produces a final answer.

Example:

detects around 230 components

detects around 5 ICs

generates complexity, pcb type, and bom estimate

The bom cost is currently generated only by the llm based on component statistics. It is not using real pricing data, so the values are approximate (for example 15000–20000 INR).

current limitations

no component classification (resistor, capacitor, etc.)

no real pricing data

ocr is still not reliable for IC text

agent still behaves similar to a fixed pipeline because tools are limited

summary

This version focuses on improving system design rather than accuracy. The pipeline is now modular and agent-based, which makes it easier to extend and improve in future.