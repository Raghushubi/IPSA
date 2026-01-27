import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    print("HF_API_KEY not found")
    exit()

with open("detection_results.json", "r") as f:
    detection_data = json.load(f)

object_type = detection_data["object_type"]
total_components = detection_data["total_components"]
component_counts = detection_data["component_counts"]

print("Object type:", object_type)
print("Total components:", total_components)
print("Component counts:")
for k, v in component_counts.items():
    print(k, ":", v)

prompt = f"""
You are given PCB component detection data from a {object_type}.

Total detected components: {total_components}

Component breakdown:
"""

for k, v in component_counts.items():
    prompt += f"- {k}: {v}\n"

prompt += """

Based on this:

1. What kind of PCB is this likely to be?
2. Is it simple, medium, or complex?
3. Give a rough BOM cost estimate in INR.

Include:
- approximate cost per category
- assembly cost
- total cost range

Return ONLY valid JSON:

{
  "pcb_type": "",
  "complexity": "",
  "confidence": 0.0,
  "bom_estimate": {
    "component_costs": {},
    "assembly_cost": 0,
    "total_cost_inr": 0,
    "cost_range_inr": ""
  },
  "reasoning": "",
  "market_reference": ""
}

Use realistic Indian electronics prices.
"""

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_API_KEY
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=800,
    temperature=0.3
)

output_text = response.choices[0].message.content

print("\nRaw LLM output:\n")
print(output_text)

# -------- JSON parsing --------

try:
    if "```json" in output_text:
        json_str = output_text.split("```json")[1].split("```")[0].strip()
    elif "```" in output_text:
        json_str = output_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = output_text.strip()

    analysis = json.loads(json_str)

except Exception:
    print("Could not parse JSON")
    analysis = {
        "raw_response": output_text
    }

with open("bom_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print("\nSaved bom_analysis.json")

if "pcb_type" in analysis:
    print("\nPCB type:", analysis.get("pcb_type"))
    print("Complexity:", analysis.get("complexity"))
    print("Confidence:", analysis.get("confidence"))
