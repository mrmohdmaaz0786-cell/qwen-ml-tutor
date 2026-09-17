import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "mohd-maaz/qwen-ml-tutor-dpo-final"

SYSTEM_PROMPT = (
    "You are an expert machine learning tutor. "
    "Answer clearly and concisely in English. "
    "Produce only one assistant answer."
)


def load_model():
    """
    Load the final merged SFT + DPO model.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    )

    model = model.to(device)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer


def chat(model, tokenizer, question):
    """
    Generate an answer for a machine-learning question.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = output[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
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
