"""
TrustGate Interactive CLI
=========================
Project: TrustGate: Model-Independent Trust Representation Learning
         for Self-Learning Large Language Models

Interactive terminal application for questioning, external verification,
comparative discrepancy analysis, and automatic self-improvement.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.trustgate_unified_pipeline import TrustGateUnifiedPipeline


def main():
    print("=" * 75)
    print("🛡️  TRUSTGATE: MODEL-INDEPENDENT TRUST REPRESENTATION LEARNING")
    print("    FOR SELF-LEARNING LARGE LANGUAGE MODELS")
    print("=" * 75)

    engine_choice = "universal"
    print("\nSelect LLM Engine:")
    print("  1. Universal Model-Independent Engine (Fast, works across all models/APIs) [Default]")
    print("  2. Local HuggingFace Qwen2.5-0.5B-Instruct Engine")
    choice = input("\nEnter choice [1 or 2, default 1]: ").strip()
    if choice == "2":
        engine_choice = "qwen"

    print(f"\nInitializing TrustGate with '{engine_choice}' engine...")
    try:
        pipeline = TrustGateUnifiedPipeline(engine_type=engine_choice, use_wikipedia=True)
    except Exception as exc:
        print(f"Warning initializing pipeline ({exc}). Falling back to Universal Engine.")
        pipeline = TrustGateUnifiedPipeline(engine_type="universal", use_wikipedia=False)

    print("\nTrustGate is ready! Commands: 'exit', 'stats', 'retrain', 'switch'")
    print("-" * 75)

    while True:
        try:
            print("\n" + "=" * 75)
            question = input("Enter your question:\n> ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                print("\nExiting TrustGate. Goodbye!")
                break

            if question.lower() == "stats":
                stats = pipeline.self_learner.get_summary_stats()
                print("\n📊 TrustGate Self-Improvement System Stats:")
                print(f"  • Total Logged Experiences:       {stats['total_experiences']}")
                print(f"  • Approved High-Trust Cycles:     {stats['approved_experiences']}")
                print(f"  • Learned Knowledge Memories:     {stats['verified_knowledge_memories']}")
                print(f"  • Successful Corrections:         {stats['successful_conversions']}")
                print(f"  • Active Model Bundle:            {stats['active_model']}")
                continue

            if question.lower() == "retrain":
                print("\nInitiating online model retraining...")
                metrics = pipeline.self_learner.trigger_model_retraining()
                pipeline.reload_model(metrics["model_path"])
                print(f"✓ Retrained model loaded! Accuracy: {metrics['accuracy']:.4f}")
                continue

            if question.lower() == "switch":
                new_engine = "universal" if pipeline.engine_type == "qwen" else "qwen"
                pipeline.set_engine(new_engine)
                print(f"✓ Switched to {pipeline.engine.model_name}")
                continue

            # Process Query
            print("\nAnalyzing question with TrustGate...")
            result = pipeline.process_query(question)

            init = result["initial"]
            init_trust = init["trustgate"]
            verification = result["verification"]
            candidate = result["candidate"]
            final_ans = result["final_answer"]
            final_lbl = result["final_label"]
            self_imp = result["self_improvement"]

            print("\n" + "-" * 75)
            print("📝 INITIAL MODEL ANSWER:")
            print(f"\"{init['answer']}\"")
            print(f"🛡️ TrustGate Assessment: [{init_trust['label']}]")
            print(f"   Confidence: {init_trust['confidence']:.4f} | Entropy: {init_trust['entropy']:.4f} | 22-D Fusion Space")

            if verification:
                print("\n🔍 EXTERNAL VERIFICATION & COMPARISON:")
                print(f"   Consensus: {verification['consensus']}")
                print(f"   Analysis:  {verification['discrepancy_explanation']}")
                if verification.get("contradictions"):
                    print(f"   ⚠️  Contradiction: {verification['contradictions'][0]}")

            if candidate:
                cand_trust = candidate["trustgate"]
                print("\n🔄 EVIDENCE-GROUNDED SELF-CORRECTION:")
                print(f"   Corrected Answer: \"{candidate['answer']}\"")
                print(f"   Upgraded Trust:   [{cand_trust['label']}] (Confidence: {cand_trust['confidence']:.4f}, Entropy: {cand_trust['entropy']:.4f})")
                print(f"   Outcome:          SUCCESSFULLY CHANGED {init_trust['label']} -> {cand_trust['label']} ✅")

            print("\n🧠 AUTONOMOUS SELF-LEARNING:")
            if self_imp["occurred"]:
                print(f"   ✓ Verified experience recorded into Trust Experience Buffer")
                print(f"   ✓ High-trust memory updated ({self_imp['total_memories']} total concepts learned)")
            else:
                print(f"   • Answer confirmed consistent with verified knowledge")

            print("\n🎯 FINAL DELIVERED ANSWER:")
            print(f"[{final_lbl}] \"{final_ans}\"")
            print("-" * 75)

        except KeyboardInterrupt:
            print("\nExiting TrustGate.")
            break
        except Exception as err:
            print(f"\n[Error] {err}")


if __name__ == "__main__":
    main()