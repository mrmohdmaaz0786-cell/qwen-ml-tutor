# Qwen ML Tutor — SFT + DPO Aligned ML Assistant

A machine learning tutor built by fine-tuning **Qwen2.5-1.5B-Instruct** using **LoRA-based Supervised Fine-Tuning (SFT)** followed by **Direct Preference Optimization (DPO)**.

## Pipeline

```text
Qwen2.5-1.5B-Instruct
        ↓
LoRA SFT
        ↓
SFT ML Tutor
        ↓
Preference Dataset
        ↓
DPO
        ↓
Final Aligned ML Tutor
```

## Project Highlights

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT dataset: **1,612 examples**
- DPO dataset: **1,036 preference pairs**
- Preference topics: **74 ML topics**
- LoRA trainable parameters: **~1.18%**
- DPO test preference accuracy: **94.6%**
- Final merged model uploaded to Hugging Face

## SFT Dataset

| Split | Examples |
|---|---:|
| Train | 1,293 |
| Validation | 161 |
| Test | 158 |
| **Total** | **1,612** |

The SFT dataset combines curated ML Q&A with a filtered subset of `tatsu-lab/alpaca`.

## DPO Preference Dataset

Each example contains a `prompt`, `chosen` answer, and `rejected` answer.

The initial dataset contained **1,184 preference pairs**. After quality checking and cleaning, **1,036 pairs** remained.

| Split | Pairs |
|---|---:|
| Train | 826 |
| Validation | 98 |
| Test | 112 |
| **Total** | **1,036** |

The split was topic-aware, so the same ML topic did not appear across train, validation, and test sets.

## LoRA Configuration

```text
Rank:            16
Alpha:           32
Dropout:         0.05
Trainable share: ~1.18%
```

Target modules:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

## DPO Configuration

```text
Epochs:                1
Learning rate:         1e-5
Beta:                  0.1
Batch size:            1
Gradient accumulation: 8
Max length:            512
Precision:             FP16
```

## Results

### SFT

```text
Best validation loss: 1.6132
Test loss:            1.6768
```

### DPO

```text
Validation loss:      0.4960
Test loss:            0.4860
Preference accuracy:  94.6%
Reward margin:        +0.502
```

> Preference accuracy measures whether the model prefers the chosen response over the rejected response. It is not factual accuracy.

## Load the Final Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mohd-maaz/qwen-ml-tutor-dpo-final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
```

## Repository Structure

```text
qwen-ml-tutor/
├── README.md
├── inference.py
├── requirements.txt
└── notebooks/
    ├── qwen_ml_tutor_training.ipynb
    └── qwen_ml_tutor_dpo_alignment.ipynb
```

## Hugging Face Model

[mohd-maaz/qwen-ml-tutor-dpo-final](https://huggingface.co/mohd-maaz/qwen-ml-tutor-dpo-final)

## Limitations

- The model has 1.5B parameters.
- The preference dataset is relatively small.
- Preference accuracy is not factual accuracy.
- Some responses may still contain imperfect wording.

## Tech Stack

Python · PyTorch · Transformers · TRL · PEFT · LoRA · Hugging Face · scikit-learn · Kaggle

## Author

**Mohd Maaz**
