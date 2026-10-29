from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

question = "What is the capital of Australia?"

# --------------------------------------------------
# STEP 1: Generate the original answer
# --------------------------------------------------

answer_prompt = f"""You are a helpful assistant.

Question:
{question}

Give the best possible answer.

Answer:
"""

inputs = tokenizer(
    answer_prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
).strip()

print("\nQUESTION:")
print(question)

print("\nLLM ANSWER:")
print(answer)


# --------------------------------------------------
# STEP 2: Ask the LLM to evaluate its own answer
# --------------------------------------------------

feedback_prompt = f"""You are evaluating an AI-generated answer.

Question:
{question}

AI Answer:
{answer}

Evaluate the answer.

Return ONLY in this format:

Confidence: <number between 0 and 1>
Self-Assessment: <Correct / Incorrect / Uncertain>
Reason: <short explanation>
"""

feedback_inputs = tokenizer(
    feedback_prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    feedback_outputs = model.generate(
        **feedback_inputs,
        max_new_tokens=150,
        do_sample=False
    )

feedback = tokenizer.decode(
    feedback_outputs[0][feedback_inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
).strip()

print("\nSELF-FEEDBACK:")
print(feedback)


# --------------------------------------------------
# STEP 3: Save the result
# --------------------------------------------------

result = {
    "question": question,
    "answer": answer,
    "self_feedback": feedback
}

with open("outputs/self_feedback.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("\nSaved to:")
print("outputs/self_feedback.json")