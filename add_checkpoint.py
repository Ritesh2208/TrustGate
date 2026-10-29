from pathlib import Path

path = Path("features/trustgate_v7_1_natural_270.py")

text = path.read_text(encoding="utf-8")

old = """            results.append(
                result
            )

            decision = result["""

new = """            results.append(
                result
            )

            # Checkpoint after every completed question
            checkpoint = {
                "experiment": "TrustGate V7.1 Natural Held-Out 270",
                "model": MODEL_NAME,
                "heldout_source": HELDOUT_FILE,
                "num_cases": NUM_QUESTIONS,
                "completed_cases": len(results),
                "results": results
            }

            with open(
                RESULT_PATH + ".checkpoint",
                "w",
                encoding="utf-8"
            ) as checkpoint_file:
                json.dump(
                    checkpoint,
                    checkpoint_file,
                    indent=2,
                    ensure_ascii=False
                )

            decision = result["""

if old not in text:
    raise RuntimeError("SUCCESS RESULT BLOCK NOT FOUND")

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("CHECKPOINT CODE ADDED")