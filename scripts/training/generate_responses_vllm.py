#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ==========================================
# CONFIGURATION
# ==========================================

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 512
BATCH_SIZE = 16  # vLLM does its own scheduling; this just chunks prompts

MODEL_BASE_NAMES = {
    "uncensored_qwen": "models/Qwen2.5-3B-Instruct-Uncensored-Test",
    "uncensored_llama": "models/Llama-3.2-3B-Instruct-uncensored",
    "finetuned_qwen": "models/Qwen2.5-3B-OpenThoughts-Merged",
    "finetuned_llama": "models/merged-llama-3.2-3b-finetuned",
}

OUTPUT_BASE_DIR = "data"

# ==========================================
# GENERATION FUNCTION
# ==========================================

def build_prompts(tokenizer, questions: List[str]) -> List[str]:
    """
    Build chat-style prompts from raw questions using the model's chat template.
    Assumes an instruct/chat model with a defined tokenizer.chat_template.
    """
    batch_messages = [[{"role": "user", "content": q}] for q in questions]

    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        for msgs in batch_messages
    ]
    return prompts


def generate_responses_vllm(llm, tokenizer, questions: List[str], output_file: str):
    """
    Generates responses for a list of questions using vLLM.
    Saves results to output_file as JSON list of {question, response}.
    """
    from tqdm import tqdm

    print(f"Generating {len(questions)} responses to {output_file}...")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )

    responses = []

    # vLLM can take a large list of prompts at once, but we still chunk to avoid silly extremes.
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Generating with vLLM"):
        batch_questions = questions[i: i + BATCH_SIZE]
        prompts = build_prompts(tokenizer, batch_questions)

        # vLLM does its own tokenization internally, so we pass strings.
        outputs = llm.generate(prompts, sampling_params)

        # Each output is a vllm.RequestOutput object
        for q, out in zip(batch_questions, outputs):
            # Take the first hypothesis
            text = out.outputs[0].text

            responses.append({
                "question": q,
                "response": text.strip()
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(responses)} generations to {output_file}")


# ==========================================
# MAIN
# ==========================================

def load_questions(input_file: str) -> List[str]:
    """
    Load questions from a JSON file.
    Supports:
      - list[str]
      - list[{"question": str}] or HC3-like structures
      - list[{"messages": [{"role": "user", "content": ...}, ...]}]
    """
    print(f"Loading questions from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict):
                if "messages" in item:
                    for msg in item["messages"]:
                        if msg.get("role") == "user":
                            questions.append(msg.get("content", ""))
                            break
                elif "question" in item:
                    questions.append(item["question"])
                elif "text" in item:
                    questions.append(item["text"])
    else:
        raise ValueError("Input JSON must be a list.")

    print(f"Loaded {len(questions)} questions")
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from local HF models using vLLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["uncensored_qwen", "uncensored_llama", "finetuned_qwen", "finetuned_llama"],
        help="Short model name (maps to MODEL_BASE_NAMES).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override: explicit path to HF model directory. "
             "If provided, --model is ignored.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input JSON file containing questions. If omitted, runs in interactive mode.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file (only used if --input-file is set).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Max sequence length for vLLM (context).",
    )

    args = parser.parse_args()

    # Determine model path
    if args.model_path is not None:
        model_path = args.model_path
    else:
        if args.model is None:
            raise ValueError("Either --model-path or --model must be provided.")
        model_path = MODEL_BASE_NAMES[args.model]

    print(f"Using model from: {model_path}")

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load vLLM engine
    llm = LLM(
        model=model_path,
        max_model_len=args.max_seq_length,
        trust_remote_code=True,  # if needed for some HF repos
    )

    # Interactive mode
    if args.input_file is None:
        print("No input file provided; entering interactive mode.")
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_NEW_TOKENS,
        )

        while True:
            try:
                user = input("\nUser: ")
            except EOFError:
                break
            if not user.strip():
                break

            # Single-turn chat message
            messages = [{"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            outputs = llm.generate([prompt], sampling_params)
            text = outputs[0].outputs[0].text.strip()

            print("\nModel:\n" + text)

    else:
        # Batch file mode
        questions = load_questions(args.input_file)

        if args.output_file is None:
            # Default output file location
            base = args.model if args.model is not None else "custom_model"
            output_dir = os.path.join(OUTPUT_BASE_DIR, base)
            os.makedirs(output_dir, exist_ok=True)
            args.output_file = os.path.join(output_dir, "vllm_generations.json")

        generate_responses_vllm(llm, tokenizer, questions, args.output_file)


if __name__ == "__main__":
    main()
