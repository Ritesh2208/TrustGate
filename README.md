# 🛡️ TrustGate: Model-Independent Trust Representation Learning for Self-Learning Large Language Models

**TrustGate** is an advanced AI reliability framework designed to detect uncertainty and unreliability in Large Language Models (LLMs), verify outputs against external empirical evidence, correct misconceptions into verified factual answers, and **autonomously self-improve** over time.

---

## 🚀 Key Innovations & Architecture

```
                                 [ USER QUERY ]
                                        │
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │            1. SELF-LEARNING KNOWLEDGE MEMORY           │
             │   Checks previously verified facts & experiences       │
             │   If found ➔ injects verified context into generator  │
             └──────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │             2. MODEL-INDEPENDENT LLM ENGINE            │
             │   Supports: Universal Model-Independent Engine (Any    │
             │   LLM/API) & Local Open-Source Models (Qwen2.5-0.5B).  │
             └──────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │       3. MODEL-INDEPENDENT TRUST REPRESENTATION        │
             │  - Extracts Behavioral Signals (Confidence, Entropy)   │
             │  - Semantic Hidden Projection (Standardized 896-D)     │
             │  - 22-D Fused Trust Space (PCA-20 + Conf + Ent)        │
             │  - TrustGate Classifier predicts:                      │
             │    🟢 RELIABLE | 🟡 UNCERTAIN | 🔴 UNRELIABLE          │
             └──────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
            Is Answer UNRELIABLE or UNCERTAIN? (or Fact-Check requested)
                                 ┌──────┴──────┐
                            YES  │             │  NO (RELIABLE & Confirmed)
                                 ▼             ▼
             ┌──────────────────────────────────────────┐     ┌──────────────┐
             │        4. EXTERNAL VERIFICATION ENGINE   │     │ Output A0    │
             │  - Multi-tier retrieval (Wikipedia API + │     │ directly     │
             │    empirical verified knowledge store)   │     └──────────────┘
             │  - Claim extraction & negation checks    │
             │  - Direct evidence quote verification    │
             │  - Consensus: SUPPORTED vs CONTRADICTED  │
             └──────────────────┬───────────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────────────────────────┐
             │      5. FACT-CHECK COMPARISON & DISCREPANCY DETECTOR   │
             │  Compares initial claims vs external verified facts.   │
             │  Identifies hallucinations, misconceptions, errors.    │
             └──────────────────┬─────────────────────────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────────────────────────┐
             │   6. EVIDENCE-GROUNDED SELF-CORRECTION (A0 -> A1)      │
             │  Synthesizes correct candidate answer A1 strictly from │
             │  verified facts (supported or contradicted facts).     │
             │  Validates A1 (no prompt leakage, meaningful change).  │
             │  Re-evaluates A1 with TrustGate & Independent Check.   │
             │  Changes Unreliable/Uncertain ➔ 🟢 RELIABLE!           │
             └──────────────────┬─────────────────────────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────────────────────────┐
             │     7. AUTONOMOUS SELF-IMPROVEMENT & CONTINUOUS LEARNING│
             │  - Automatic Buffer Storage: saves verified experience │
             │  - High-Trust Memory Bank: updates instant recall store│
             │  - Model Evolution Retraining: when experiences        │
             │    accumulate, auto-retrains 22-D classifier and       │
             │    hot-swaps model bundle with rollback protection     │
             │  - Next time ANY user asks a related question, LLM     │
             │    answers correctly on the first attempt!             │
             └────────────────────────────────────────────────────────┘
```

---

## 🌟 Highlights

1. **Model Independence**:
   - Operates with **any** LLM architecture (OpenAI, Anthropic, Gemini, Llama, Mistral, Qwen, or custom local models).
   - Maps arbitrary model outputs into a unified **22-D Trust Representation Space** (20 PCA components + Confidence + Entropy).

2. **Comparative External Verification & Discrepancy Detection**:
   - When an answer is `UNRELIABLE` or `UNCERTAIN`, TrustGate retrieves empirical evidence from Wikipedia and verified scientific sources.
   - Compares the claim against evidence, identifying factual contradictions (e.g. chemical misconceptions like *"bleach is an acid"* vs verified fact *"bleach is alkaline sodium hypochlorite"*).

3. **Evidence-Grounded Self-Correction**:
   - Synthesizes a factual, evidence-grounded candidate ($A_1$) replacing the unreliable response.
   - Re-evaluates $A_1$ through TrustGate to confirm trust upgrade to `RELIABLE`.

4. **Autonomous Self-Improvement**:
   - **Continuous Memory Integration**: High-trust corrections are added to `outputs/trust_learned_knowledge.json`. Future queries leverage this memory to produce correct answers immediately.
   - **Online Model Retraining**: Experiences accumulate in `outputs/trust_experience_buffer.json`. When the threshold is reached (or on demand), the 22-D LinearSVC classifier auto-retrains on baseline human data + verified experiences, hot-reloading `models/trustgate_self_improved.joblib`.

---

## 🛠️ Project Structure

```text
TrustGate/
├── app.py                                   # Streamlit Web Application Dashboard
├── run_trustgate.py                         # Interactive Terminal CLI
├── run_trustgate_auto.py                    # Automated Benchmark & Self-Improvement Suite
├── features/
│   ├── trustgate_model_engine.py            # Model-Independent LLM Engine Interface
│   ├── trustgate_knowledge_memory.py        # Self-Learning Knowledge Memory Bank
│   ├── trustgate_enhanced_verifier.py       # Multi-Tier Verification & Comparison Engine
│   ├── trustgate_self_learner.py            # Continuous Retraining & Experience Controller
│   ├── trustgate_unified_pipeline.py        # Master TrustGate End-to-End Orchestrator
│   ├── trustgate_controller.py              # 22-D Representation & LinearSVC Classifier
│   ├── external_verifier.py                 # MediaWiki Wikipedia Retrieval
│   └── trust_experience_buffer.py           # Atomic Experience Buffer Persistence
├── models/
│   ├── trustgate_500_22d_linear_svm.joblib  # Baseline 500-sample 22-D TrustGate Model
│   ├── trustgate_self_improved.joblib       # Online Self-Improved Retrained Model
│   └── trustgate_v2_self_trained.joblib     # Self-Trained Checkpoint Bundle
└── outputs/
    ├── trust_experience_buffer.json         # Logged Self-Improvement Experiences
    ├── trust_learned_knowledge.json         # Self-Learned Knowledge Base
    └── trustgate_benchmark_results.json     # Automated Evaluation Metrics
```

---

## 💻 How to Run

Activate the project virtual environment before running:

### 1. Interactive Web Application (Streamlit)
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
- Switch between Universal Model-Independent Engine and Local Qwen.
- Click sample query chips or type your own question.
- Inspect the Initial TrustGate evaluation, External Evidence comparison, and Evidence-Grounded Correction.
- Watch the Self-Learning Hub update dynamically with new memories and experiences.
- Trigger model self-evolution on demand.

### 2. Interactive Terminal CLI
```powershell
.\.venv\Scripts\python.exe run_trustgate.py
```
Provides an interactive command prompt to ask questions, view comparative analysis, inspect trust classifications, and monitor self-learning.

### 3. Automated Benchmark & Verification Suite
```powershell
.\.venv\Scripts\python.exe run_trustgate_auto.py
```
Executes an end-to-end benchmark across core misconception challenges (chemistry, longevity, human physiology, astronomy, geography), verifying that:
- Every `UNRELIABLE`/`UNCERTAIN` answer is converted to `RELIABLE`.
- Experiences are logged and memory is populated.
- Online model retraining executes automatically.
