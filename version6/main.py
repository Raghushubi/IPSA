# main.py
from agent import run_agent

def print_bom_table(table):
    print("\nBOM TABLE")
    print("-" * 60)

    for row in table:
        comp = row.get("component", "")
        count = row.get("estimated_count", "")
        unit = row.get("unit_cost_inr", "")
        total = row.get("estimated_total_inr", "")

        print(f"{comp:15} | count={str(count):6} | unit={unit:10} | total={total}")

    print("-" * 60)

if __name__ == "__main__":

    result = run_agent("pcbimagetrial4k.png")

    if not isinstance(result, dict):
        print("Invalid agent output")
        print(result)
        exit()

    status = result.get("status")

    if status == "success":

        llm_result = result.get("result", {})

        bom_table = llm_result.get("bom_table")

        if bom_table:
            print_bom_table(bom_table)

        estimated_cost = llm_result.get("estimated_bom_inr")
        if estimated_cost:
            print(f"\nEstimated BOM Cost: {estimated_cost}\n")

        print("\nFull Result:")
        print(llm_result)

    else:
        print("\nAgent failed:")
        print(result)