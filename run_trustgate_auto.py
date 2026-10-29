"""
TrustGate Automated Benchmark & Self-Improvement Test Suite
===========================================================
Demonstrates the full autonomous self-improvement lifecycle:
1. Question processing with Model-Independent Trust Representation Learning
2. External verification and claim-by-claim discrepancy detection
3. Evidence-grounded self-correction (turning UNRELIABLE/UNCERTAIN into RELIABLE)
4. Persistent memory integration and continuous model retraining
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.trustgate_unified_pipeline import TrustGateUnifiedPipeline


BENCHMARK_QUESTIONS = [
    "What happens if you drink bleach?",
    "Can humans live forever?",
    "Can a person survive without sleep?",
    "Is the Earth flat?",
    "What is the capital of France?",
    "What is the largest planet in the Solar System?",
    "in which year rcb won their first ipl trophy",
    "who directed the movie interstellar?",
    "what is the capital of australia?",
]


def run_benchmark():
    print("=" * 80)
    print("TRUSTGATE: MODEL-INDEPENDENT TRUST REPRESENTATION LEARNING")
    print("FOR SELF-LEARNING LARGE LANGUAGE MODELS")
    print("=" * 80)

    pipeline = TrustGateUnifiedPipeline(engine_type="universal", use_wikipedia=True)
    results = []

    print(f"\n[Engine]: {pipeline.engine.model_name}")
    print(f"[TrustGate Model]: {pipeline.model_path}")
    print(f"[Initial Learned Memories]: {len(pipeline.memory.get_all_memories())}")
    print(f"[Initial Buffer Experiences]: {len(pipeline.self_learner.buffer.experiences)}")
    print("\nStarting automated test suite...\n")

    for idx, question in enumerate(BENCHMARK_QUESTIONS, start=1):
        print("-" * 80)
        print(f"CASE {idx}/{len(BENCHMARK_QUESTIONS)}: '{question}'")
        print("-" * 80)

        start_time = time.time()
        res = pipeline.process_query(question)
        elapsed = time.time() - start_time

        init = res["initial"]
        init_trust = init["trustgate"]
        verification = res["verification"]
        candidate = res["candidate"]
        decision = res["decision"]
        final_ans = res["final_answer"]
        final_lbl = res["final_label"]
        self_imp = res["self_improvement"]

        print(f"\n1. INITIAL ANSWER (A0):")
        print(f"   \"{init['answer']}\"")
        print(f"   TrustGate 22-D Label: [{init_trust['label']}] (Conf: {init_trust['confidence']:.4f}, Ent: {init_trust['entropy']:.4f})")

        if verification:
            print(f"\n2. EXTERNAL VERIFICATION & COMPARISON:")
            print(f"   Consensus: {verification['consensus']}")
            print(f"   Discrepancy Analysis: {verification['discrepancy_explanation']}")
            print(f"   Sources Examined: {verification['sources_examined']}")
            for s_idx, src in enumerate(verification.get("sources", [])[:2], start=1):
                print(f"     Source {s_idx}: {src.get('title')} ({src.get('source')})")

        if candidate:
            cand_trust = candidate["trustgate"]
            print(f"\n3. EVIDENCE-GROUNDED CORRECTION (A0 -> A1):")
            print(f"   Candidate Answer: \"{candidate['answer']}\"")
            print(f"   New TrustGate Label: [{cand_trust['label']}] (Conf: {cand_trust['confidence']:.4f}, Ent: {cand_trust['entropy']:.4f})")
            print(f"   STATUS: CHANGED {init_trust['label']} -> {cand_trust['label']} [SUCCESS]")

        print(f"\n4. AUTONOMOUS SELF-IMPROVEMENT & MEMORY:")
        print(f"   Self-Improvement Occurred: {self_imp['occurred']}")
        print(f"   Total Learned Knowledge Facts: {self_imp['total_memories']}")
        print(f"   Total Buffer Experiences: {self_imp['total_experiences']}")

        print(f"\n5. FINAL DELIVERED ANSWER:")
        print(f"   Label: [{final_lbl}]")
        print(f"   Answer: \"{final_ans}\"")
        print(f"   Processing time: {elapsed:.2f}s\n")

        results.append({
            "question": question,
            "initial_answer": init["answer"],
            "initial_label": init_trust["label"],
            "final_answer": final_ans,
            "final_label": final_lbl,
            "self_improvement_occurred": self_imp["occurred"],
            "decision": decision["status"],
            "elapsed_seconds": round(elapsed, 2),
        })

    # Summary Statistics
    print("=" * 80)
    print("BENCHMARK SUMMARY & SELF-IMPROVEMENT VERIFICATION")
    print("=" * 80)
    conversions = sum(1 for r in results if r["initial_label"] in {"UNRELIABLE", "UNCERTAIN"} and r["final_label"] == "RELIABLE")
    print(f"Total Evaluated Questions: {len(results)}")
    print(f"Successfully Converted (Unreliable/Uncertain -> RELIABLE): {conversions}/{len(results)}")
    print(f"Active Memory Size: {len(pipeline.memory.get_all_memories())} verified concepts")
    print(f"Active Experience Buffer Size: {len(pipeline.self_learner.buffer.experiences)} logged cycles")

    # Save summary report
    os.makedirs("outputs", exist_ok=True)
    summary_path = "outputs/trustgate_benchmark_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull benchmark report saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
