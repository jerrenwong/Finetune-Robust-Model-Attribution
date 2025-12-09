import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import time

# ==========================================
# CONFIGURATION
# ==========================================

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to use (excluding qwen - use generate_responses_qwen_local.py for qwen)
MODELS = {
    "llama": "meta-llama/llama-3.2-3b-instruct",
    "gemma": "google/gemma-3-4b-it",
    "ministral": "mistralai/ministral-3b",
}

# HC3 questions paths
TRAIN_QUESTIONS_PATH = "data/hc3_questions_train.json"
TEST_QUESTIONS_PATH = "data/hc3_questions_test.json"

# Output directory
OUTPUT_BASE_DIR = "data"

# Number of questions to generate (2500)
NUM_QUESTIONS = 2500

# Number of generations per question
NUM_GENERATIONS_PER_QUESTION = 10

# Parallelization (10 workers)
NUM_WORKERS = 10

# Generation parameters
MAX_TOKENS = 512
TEMPERATURE = 0.7

# Periodic save interval (save every N questions completed)
SAVE_INTERVAL_QUESTIONS = 1000
# Also save every N seconds as backup
SAVE_INTERVAL_SECONDS = 300

# ==========================================
# GENERATION FUNCTIONS
# ==========================================

def generate_single_response(question, model_id, api_key):
    """
    Generate a single response using OpenRouter API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",  # Optional: for OpenRouter tracking
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Extract the generated text
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text.strip()
        else:
            print(f"Unexpected API response format: {result}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def save_responses(responses, output_file):
    """
    Save responses to file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving responses: {e}")
        return False


def generate_responses_for_model(model_name, model_id, questions, api_key, output_file, num_generations_per_question):
    """
    Generate multiple responses for each question using parallel workers.
    Saves periodically to avoid data loss.
    """
    print(f"\nGenerating responses for {model_name} ({model_id})...")
    print(f"Total questions: {len(questions)}")
    print(f"Generations per question: {num_generations_per_question}")
    print(f"Using {NUM_WORKERS} parallel workers")
    print(f"Saving every {SAVE_INTERVAL_QUESTIONS} questions or every {SAVE_INTERVAL_SECONDS} seconds")

    # Initialize responses dict: question -> list of responses
    responses_dict = {q: [] for q in questions}
    completed_questions = 0
    failed_count = 0
    last_save_time = time.time()
    last_saved_question_count = 0

    # Create all tasks: (question, generation_index)
    tasks = []
    for question in questions:
        for gen_idx in range(num_generations_per_question):
            tasks.append((question, gen_idx))

    total_tasks = len(tasks)
    print(f"Total API calls: {total_tasks}")

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_single_response, question, model_id, api_key): (question, gen_idx)
            for question, gen_idx in tasks
        }

        # Process completed tasks with progress bar
        with tqdm(total=total_tasks, desc=f"Generating {model_name}") as pbar:
            for future in as_completed(future_to_task):
                question, gen_idx = future_to_task[future]
                try:
                    generated_text = future.result()
                    if generated_text:
                        responses_dict[question].append(generated_text)
                    else:
                        failed_count += 1
                        print(f"\nFailed to generate response {gen_idx+1} for question: {question[:50]}...")
                except Exception as e:
                    failed_count += 1
                    print(f"\nException generating response: {e}")
                finally:
                    pbar.update(1)

                # Check if question is complete (all generations done)
                question_just_completed = False
                if len(responses_dict[question]) == num_generations_per_question:
                    completed_questions += 1
                    question_just_completed = True

                # Periodic save: every N questions or every N seconds
                current_time = time.time()
                should_save = False

                # Save if we've completed N more questions since last save
                if question_just_completed and (completed_questions - last_saved_question_count) >= SAVE_INTERVAL_QUESTIONS:
                    should_save = True
                # Or save if enough time has passed
                elif current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
                    should_save = True

                if should_save:
                    # Convert dict to list format for saving
                    responses_list = [
                        {
                            "question": q,
                            "responses": responses_dict[q]
                        }
                        for q in questions if len(responses_dict[q]) > 0
                    ]
                    if save_responses(responses_list, output_file):
                        print(f"\n[Checkpoint] Saved {completed_questions}/{len(questions)} questions completed")
                    last_save_time = current_time
                    last_saved_question_count = completed_questions

    # Final save
    responses_list = [
        {
            "question": q,
            "responses": responses_dict[q]
        }
        for q in questions
    ]

    total_responses = sum(len(responses_dict[q]) for q in questions)
    print(f"\nCompleted: {completed_questions}/{len(questions)} questions, {total_responses} total responses, {failed_count} failed")

    if save_responses(responses_list, output_file):
        print(f"Saved final results to {output_file}")

    return completed_questions, total_responses, failed_count


def main():
    parser = argparse.ArgumentParser(description="Generate responses using OpenRouter API.")
    parser.add_argument("--api-key", type=str, help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS,
                       help=f"Number of questions to generate (default: {NUM_QUESTIONS})")
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS_PER_QUESTION,
                       help=f"Number of generations per question (default: {NUM_GENERATIONS_PER_QUESTION})")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                       help=f"Number of parallel workers (default: {NUM_WORKERS})")
    parser.add_argument("--dataset", type=str, choices=["train", "test"], default="train",
                       help="Dataset to generate for: 'train' or 'test' (default: 'train')")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()) + ["all"], default="all",
                       help="Model to generate for, or 'all' for all models (default: 'all')")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: only generate for first 10 questions")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key required. Provide via --api-key or OPENROUTER_API_KEY env var.")
        return

    # Determine which dataset to use
    if args.dataset == "test":
        questions_path = TEST_QUESTIONS_PATH
        dataset_prefix = "hc3_test"
    else:
        questions_path = TRAIN_QUESTIONS_PATH
        dataset_prefix = "hc3_train"

    # Load HC3 questions
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
    except Exception as e:
        print(f"Error loading questions from {questions_path}: {e}")
        return

    # Determine number of questions to use
    if args.test:
        num_questions_to_use = 10
        print("TEST MODE: Using first 10 questions only")
    else:
        num_questions_to_use = args.num_questions

    # Take first N questions
    questions = all_questions[:num_questions_to_use]
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Determine which models to generate for
    if args.model == "all":
        models_to_generate = MODELS
    else:
        models_to_generate = {args.model: MODELS[args.model]}

    # Generate responses for each model
    results = {}
    for model_name, model_id in models_to_generate.items():
        output_dir = os.path.join(OUTPUT_BASE_DIR, model_name, "hc3_openrouter")
        if args.test:
            output_file = os.path.join(output_dir, f"{dataset_prefix}_generations_test.json")
        else:
            output_file = os.path.join(output_dir, f"{dataset_prefix}_generations.json")

        completed_questions, total_responses, fail_count = generate_responses_for_model(
            model_name, model_id, questions, api_key, output_file, args.num_generations
        )

        results[model_name] = {
            "completed_questions": completed_questions,
            "total_responses": total_responses,
            "failed": fail_count,
            "output_file": output_file
        }

        # Small delay between models to avoid rate limits
        time.sleep(2)

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Completed Questions: {result['completed_questions']}/{len(questions)}")
        print(f"  Total Responses: {result['total_responses']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Output: {result['output_file']}")
    print("="*60)


if __name__ == "__main__":
    main()
