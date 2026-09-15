import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "mohd-maaz/qwen-ml-tutor-final"

SYSTEM_PROMPT = (
    "You are an expert machine learning tutor. "
    "Answer clearly and concisely in English. "
    "Produce only one assistant answer."
)


def load_model():
    """
    Load the original Qwen model and attach the fine-tuned LoRA adapter.
    """

    # Needed only if your Hugging Face adapter repository is private.
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_ID,
        token=hf_token
    )

    # Use GPU when available
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
        token=hf_token
    )

    model.eval()
    model.config.use_cache = True

    # Greedy generation settings
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return model, tokenizer


def chat(model, tokenizer, question):
    """
    Generate an answer for a machine-learning question.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Put inputs on the same device as the model embeddings
    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer


if __name__ == "__main__":

    print("Loading ML Tutor model...")

    model, tokenizer = load_model()

    print("Model loaded successfully.\n")

    question = "What is the difference between precision and recall?"

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(chat(model, tokenizer, question))
