# Qwen ML Tutor — LoRA Fine-Tuned Machine Learning Assistant

A lightweight machine learning tutor built by fine-tuning **Qwen2.5-1.5B-Instruct** with **LoRA (Low-Rank Adaptation)** on a curated machine learning question-answer dataset.

The project focuses on producing short, clear, beginner-friendly answers to common ML questions while keeping training efficient enough to run on a Kaggle GPU.

---

## Project Overview

This project fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` to answer machine learning questions such as:

- What is machine learning?
- What is the difference between precision and recall?
- When should lasso regression be used instead of ridge regression?
- What is the difference between a validation set and a test set?
- Why should the test set not be used for hyperparameter tuning?

The final model is trained as a **single LoRA adapter** on top of the Qwen Instruct base model.

### Final model setup

```text
Qwen/Qwen2.5-1.5B-Instruct
            +
mohd-maaz/qwen-ml-tutor-final
        LoRA adapter
            ↓
      ML Tutor Model
```

---

## Why This Project

The main goal was not only to fine-tune a language model, but to build a cleaner and more reliable training pipeline.

During development, the pipeline was improved by:

- switching from the Qwen base model to the instruction-tuned checkpoint,
- removing an unnecessary two-stage adapter workflow,
- using one clean LoRA training stage,
- using group-aware train/validation/test splitting,
- filtering irrelevant Alpaca examples,
- removing ambiguous or conceptually weak examples,
- checking the actual ChatML formatting before training,
- evaluating on a held-out test set,
- and reloading the final adapter from Hugging Face to verify that it works independently of the training session.

---

## Base Model

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Training method:** Supervised Fine-Tuning (SFT)
- **Parameter-efficient method:** LoRA
- **Frameworks:** Hugging Face Transformers, TRL, PEFT, Datasets

---

## Dataset

The final dataset combines:

1. A curated ML question-answer dataset derived from machine learning study material.
2. A filtered subset of `tatsu-lab/alpaca` containing ML-related instructions.

The book-derived Q&A data is not redistributed in this repository.

### Cleaning performed

The dataset pipeline removes:

- duplicate normalized questions,
- very short and excessively long answers,
- selected non-English-script content,
- irrelevant Alpaca prompts that matched ML keywords accidentally,
- and several conceptually weak examples related to validation/testing terminology.

One important cleaning issue came from the word **"recall"**. A naive keyword filter incorrectly selected prompts such as recalling dog breeds, quotes, memories, and historical facts. The filter was updated to use ML-specific phrases such as:

```text
precision-recall
precision score
recall score
true positive
false positive
false negative
classification metric
F1-score
```

instead of accepting every occurrence of the standalone word `recall`.

### Final dataset size

| Split | Examples |
|---|---:|
| Train | 1,293 |
| Validation | 161 |
| Test | 158 |
| **Total** | **1,612** |

### Group-aware splitting

Related book questions were grouped by concept before splitting. This prevents closely related questions from the same concept from being scattered across train and test data.

```text
Train       ≈ 80%
Validation  ≈ 10%
Test        ≈ 10%
```

---

## Chat Format

The model is trained using Qwen's ChatML-style conversation format.

Example:

```text
<|im_start|>system
You are an expert machine learning tutor. Answer clearly and concisely in English. Produce only one assistant answer.<|im_end|>
<|im_start|>user
What is machine learning?<|im_end|>
<|im_start|>assistant
Machine learning builds programs that learn useful behavior from examples instead of requiring every decision rule to be written by hand...<|im_end|>
```

The dataset is passed to TRL as conversational `prompt` and `completion` fields so that training loss is calculated on the assistant completion only.

---

## LoRA Configuration

```python
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
```

Trainable parameters during LoRA fine-tuning:

```text
18,464,768 trainable parameters
1,562,179,072 total parameters
≈ 1.18% trainable
```

---

## Training Configuration

Key training settings:

| Parameter | Value |
|---|---:|
| Epochs | 3 maximum |
| Train batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Eval batch size | 2 |
| Learning rate | `1e-4` |
| Scheduler | Cosine |
| Warmup ratio | `0.1` |
| Weight decay | `0.01` |
| Max sequence length | 512 |
| Max gradient norm | `1.0` |
| Precision | FP16 |
| Gradient checkpointing | Enabled |
| Completion-only loss | Enabled |
| Packing | Disabled |
| Early stopping patience | 1 |

The trainer restores the checkpoint with the lowest validation loss using:

```python
load_best_model_at_end=True
metric_for_best_model="eval_loss"
greater_is_better=False
```

---

## Results

Final run:

```text
Best checkpoint:
/kaggle/working/qwen-ml-tutor-final-clean/checkpoint-161

Best validation loss:
1.6132

Test loss:
1.6768
```

The test loss is close to the best validation loss, which is a useful signal that performance did not collapse on the held-out split. These loss values are project-specific and are not directly comparable to standardized benchmark scores.

---

## Example Model Outputs

### Precision vs Recall

**Question**

```text
What is the difference between precision and recall?
```

**Answer**

```text
Precision measures how many of a model's positive predictions were actually correct, while recall measures how many of the actual positives were correctly identified by the model. Precision focuses on reducing false positives, while recall focuses on reducing false negatives.
```

### Lasso vs Ridge

**Question**

```text
When should I use lasso regression instead of ridge regression?
```

**Answer**

```text
Use lasso when a sparse solution is important, such as for feature selection or to reduce overfitting. Lasso can remove some features entirely while keeping others, whereas ridge keeps all coefficients nonzero but with smaller values.
```

### Validation vs Test Set

**Question**

```text
What is the difference between a validation set and a test set?
```

**Answer**

```text
A validation set is used to tune hyperparameters, while a test set is used to evaluate how well the model generalizes on unseen data. The validation set should be kept separate from the training and test sets so that it can be used for tuning without affecting the final evaluation.
```

---

## Installation

A compatible environment used during development included:

```bash
pip install \
  transformers==4.51.3 \
  datasets==3.6.0 \
  accelerate==1.7.0 \
  peft==0.15.2 \
  trl==0.17.0
```

You also need PyTorch with GPU support for practical inference/training performance.

---

## Load the Fine-Tuned Model

The Hugging Face repository stores the LoRA adapter, not a duplicated copy of the complete 1.5B base model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "mohd-maaz/qwen-ml-tutor-final"

# If the adapter repository is private, authenticate with Hugging Face first.
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
)

model.eval()
model.config.use_cache = True

# Avoid generation warnings when using greedy decoding.
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
```

If the adapter repository is private, pass a Hugging Face token or log in before loading.

---

## Inference Example

```python
import torch

SYSTEM = (
    "You are an expert machine learning tutor. "
    "Answer clearly and concisely in English. "
    "Produce only one assistant answer."
)


def chat(question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.get_input_embeddings().weight.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


print(chat("What is the difference between precision and recall?"))
```

---

## Verification Performed

The final adapter was uploaded to Hugging Face and then loaded again on top of a fresh copy of `Qwen/Qwen2.5-1.5B-Instruct`.

A fresh inference test after reloading returned a correct precision-vs-recall explanation, confirming that the saved adapter can be reused independently of the original training session.

---

## What Went Wrong During Development — and What Was Fixed

This project also involved debugging several real fine-tuning issues:

### 1. Base model instead of Instruct model

An early version used `Qwen/Qwen2.5-1.5B`. The final pipeline uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

which is a better starting point for conversational instruction following.

### 2. Multiple-adapter workflow

An earlier experiment merged a previous LoRA adapter and then attached a second adapter. This produced PEFT warnings and made the training state harder to reason about.

The final version uses:

```text
Fresh Qwen Instruct model
        +
One LoRA adapter
```

### 3. Irrelevant keyword matches

The word `recall` matched general English prompts unrelated to the ML metric. The filtering logic was made more specific.

### 4. Weak evaluation examples

Several ambiguous Alpaca rows involving cross-validation, validation, and test-set terminology were removed before the final training run.

### 5. Generation warnings

The model was decoded greedily with `do_sample=False`, so sampling-only generation settings such as `temperature`, `top_p`, and `top_k` were unset during final inference.

---

## Limitations

- The model is small (`1.5B`) and is not expected to outperform larger general-purpose LLMs.
- Evaluation currently focuses on held-out loss and manual question-answer checks rather than a large standardized ML benchmark.
- A few answers can still use imperfect wording even when the core concept is correct.
- The training dataset is relatively small.
- The model is intended as an educational ML tutor, not as a source for high-stakes decisions.

---

## Future Improvements

Possible next steps:

- create a larger manually verified ML evaluation set,
- add automatic metrics and LLM-as-a-judge evaluation,
- compare the fine-tuned model against the untouched Qwen Instruct baseline,
- evaluate hallucination and factual consistency,
- add a lightweight Gradio or Streamlit chat interface,
- deploy the tutor as a small demo application,
- and add experiment tracking for future training runs.

---

## Resume Description

**ML Tutor — Qwen2.5 LoRA Fine-Tuning**  
Fine-tuned `Qwen2.5-1.5B-Instruct` using LoRA and TRL on a curated 1,612-example machine learning Q&A dataset. Built a group-aware train/validation/test pipeline, cleaned noisy Alpaca examples, trained only ~1.18% of model parameters, evaluated the final model on a held-out test set, and verified the saved Hugging Face adapter through fresh reload inference.

### Short resume bullet version

```text
Fine-tuned Qwen2.5-1.5B-Instruct with LoRA on a curated 1.6K-example ML Q&A dataset; implemented group-aware splitting, data-quality filtering, held-out evaluation, and reusable Hugging Face adapter deployment using TRL/PEFT.
```

---

## Repository

Fine-tuned LoRA adapter:

```text
mohd-maaz/qwen-ml-tutor-final
```

> The repository may require authentication if it is kept private on Hugging Face.

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL
- PEFT / LoRA
- scikit-learn
- Kaggle GPU
- Hugging Face Hub

---

## Author

**Mohd Maaz**

AI/ML learner building practical projects in machine learning, LLM fine-tuning, evaluation, and AI engineering.
