import os
import json
import random
from datasets import load_dataset

def download_hc3_questions(output_dir, num_questions=6000, train_size=5000, seed=42):
    # Load dataset directly from URL
    dataset = load_dataset("json", data_files="https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl")
    data = list(dataset['train'])

    # Shuffle and extract unique questions
    random.seed(seed)
    random.shuffle(data)

    questions = []
    seen = set()

    for item in data:
        q = item.get('question', '').strip()
        if q and q not in seen:
            questions.append(q)
            seen.add(q)
        if len(questions) >= num_questions:
            break

    # Split into train and test
    train_questions = questions[:train_size]
    test_questions = questions[train_size:]

    # Save to files
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'hc3_questions_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_questions, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'hc3_questions_test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "data")
    download_hc3_questions(output_dir)
