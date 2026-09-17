# Qwen ML Tutor — SFT + DPO Aligned Machine Learning Assistant

A lightweight machine learning tutor built by fine-tuning **Qwen2.5-1.5B-Instruct** with **LoRA-based Supervised Fine-Tuning (SFT)** followed by **Direct Preference Optimization (DPO)**.

The goal is to produce short, clear, beginner-friendly answers to common machine learning questions while demonstrating an end-to-end LLM fine-tuning and alignment pipeline.

## Pipeline

```text
Qwen2.5-1.5B-Instruct
        ↓
LoRA Supervised Fine-Tuning
        ↓
SFT ML Tutor
        ↓
Preference Dataset Curation
        ↓
Direct Preference Optimization
        ↓
Final Preference-Aligned ML Tutor
```

## Project Highlights

- Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` using LoRA.
- Built and cleaned a **1,612-example SFT dataset**.
- Used group-aware train/validation/test splitting to reduce concept leakage.
- Created and audited a **1,184-pair preference dataset** for DPO.
- Removed template-heavy and weak preference pairs, leaving **1,036 high-quality pairs**.
- Used topic-aware DPO splits with **0 topic overlap** between train, validation, and test.
- Trained only about **1.18% of model parameters** with LoRA.
- Achieved **94.6% preference accuracy** on the held-out DPO test set.
- Merged SFT and DPO learning into a final standalone model.
- Published the final model on Hugging Face.

## SFT Dataset

The supervised fine-tuning dataset combines:

1. Curated machine learning Q&A derived from study material.
2. A filtered subset of `tatsu-lab/alpaca` containing ML-related instructions.

The book-derived Q&A data is not redistributed in this repository.

| Split | Examples |
|---|---:|
| Train | 1,293 |
| Validation | 161 |
| Test | 158 |
| **Total** | **1,612** |

The dataset was cleaned for duplicates, weak examples, irrelevant keyword matches, answer length, and selected non-English-script content.

## Preference Dataset for DPO

Each DPO example contains:

- `prompt`
- `chosen` — preferred answer
- `rejected` — weaker or incorrect answer

The initial dataset contained **1,184 preference pairs across 74 ML topics**.

After auditing answer lengths, textual similarity, repeated templates, and pair quality, weak `parameter` and `workflow` examples were removed and selected `limitation` and `what_if` examples were cleaned.

| Split | Preference Pairs |
|---|---:|
| Train | 826 |
| Validation | 98 |
| Test | 112 |
| **Total** | **1,036** |

Topic-aware splitting produced:

```text
Train ∩ Validation = 0 topics
Train ∩ Test       = 0 topics
Validation ∩ Test  = 0 topics
```

## LoRA Configuration

```text
Rank (r)        = 16
Alpha           = 32
Dropout         = 0.05
Trainable       ≈ 18.46M parameters
Total           ≈ 1.56B parameters
Trainable share ≈ 1.18%
```

Target modules:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

## DPO Configuration

```text
Epochs                    = 1
Learning rate             = 1e-5
Beta                      = 0.1
Train batch size          = 1
Gradient accumulation     = 8
Effective batch size      = 8
Max sequence length       = 512
Precision                 = FP16
Gradient checkpointing    = Enabled
```

The SFT LoRA was first merged into the base model. A fresh LoRA adapter was then trained with DPO and finally merged again to produce the standalone aligned model.

## Results

### SFT

```text
Best validation loss: 1.6132
Test loss:            1.6768
```

### DPO

```text
Validation loss:       0.4960
Test loss:             0.4860
Test preference acc.:  94.6%
Chosen reward:        +0.173
Rejected reward:      -0.329
Reward margin:        +0.502
```

> **Preference accuracy** means the model assigned a higher preference score to the chosen answer than to the rejected answer. It is not the same as factual accuracy.

## Example Output

**Question**

```text
What is the difference between precision and recall?
```

**Answer**

```text
Precision measures how many of a model's positive predictions were actually correct,
while recall measures how many of the actual positives were correctly identified by
the model. Precision focuses on reducing false positives, while recall focuses on
reducing false negatives.
```

## Load the Final Model

The final SFT + DPO model is fully merged, so no separate PEFT adapter is required for inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mohd-maaz/qwen-ml-tutor-dpo-final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
```

## Inference

```python
messages = [
    {
        "role": "system",
        "content": "You are an expert machine learning tutor. Answer clearly and concisely in English."
    },
    {
        "role": "user",
        "content": "What is the difference between precision and recall?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    repetition_penalty=1.1,
)

input_length = inputs["input_ids"].shape[-1]

answer = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True,
)

print(answer)
```

## Repository Structure

```text
qwen-ml-tutor/
├── README.md
├── inference.py
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── qwen_ml_tutor_training.ipynb
    └── qwen_ml_tutor_dpo_alignment.ipynb
```

## Model

Final aligned model:

```text
mohd-maaz/qwen-ml-tutor-dpo-final
```

SFT adapter:

```text
mohd-maaz/qwen-ml-tutor-final
```

## Limitations

- The model is relatively small at 1.5B parameters.
- The preference dataset is synthetic/curated and not a large human-annotated preference benchmark.
- Preference accuracy does not measure general factual accuracy.
- Some responses can still contain imperfect wording.
- The model is intended for educational ML tutoring, not high-stakes decisions.

## Future Improvements

- Build a larger manually verified ML evaluation set.
- Compare the aligned model against the untouched Qwen baseline.
- Add factuality and hallucination evaluation.
- Add LLM-as-a-judge and automatic evaluation metrics.
- Build a lightweight Gradio or Streamlit demo.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Datasets · TRL · PEFT/LoRA · scikit-learn · Kaggle GPU · Hugging Face Hub

## Resume Description

**Qwen ML Tutor — SFT + DPO Alignment**

Fine-tuned `Qwen2.5-1.5B-Instruct` with LoRA-based SFT and DPO using curated ML instruction and preference datasets. Built leakage-aware data splits, audited preference quality, trained only ~1.18% of parameters, achieved **94.6% preference accuracy** on a topic-held-out DPO test set, and published the final merged model on Hugging Face.

## Author

**Mohd Maaz**

AI/ML learner building practical projects in machine learning, LLM fine-tuning, evaluation, and AI engineering.
