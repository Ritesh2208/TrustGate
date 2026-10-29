import json

with open(
    "outputs/trustgate_self_improvement_results.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

for item in data:
    print("\nQ:", item["question"])

    for h in item["history"]:
        print(
            " Iteration", h["iteration"],
            "(", h["stage"], "):",
            h["trust_label"],
            " confidence=", round(h["confidence"], 4),
            " entropy=", round(h["entropy"], 4)
        )