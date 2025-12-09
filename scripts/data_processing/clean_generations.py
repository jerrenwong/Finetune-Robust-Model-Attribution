import os
import json
import glob

def clean_file(filepath):
    print(f"Cleaning {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    for item in data:
        original_response = item.get("response", "")
        cleaned_response = original_response

        if "assistant\n" in original_response:
            cleaned_response = original_response.split("assistant\n")[-1]
        elif "model\n" in original_response:
            cleaned_response = original_response.split("model\n")[-1]
        elif "assistant" in original_response:
            parts = original_response.split("assistant")
            if len(parts) > 1:
                cleaned_response = parts[-1]
        elif "model" in original_response:
            parts = original_response.split("model")
            if len(parts) > 1:
                cleaned_response = parts[-1]

        cleaned_response = cleaned_response.strip()

        if cleaned_response != original_response:
            item["response"] = cleaned_response
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved cleaned file: {filepath}")
    else:
        print(f"No changes needed for {filepath}")

def main():
    base_dir = "data"
    files = glob.glob(os.path.join(base_dir, "*", "*", "generations_step_*.json"))

    print(f"Found {len(files)} files to process.")

    for filepath in files:
        clean_file(filepath)

if __name__ == "__main__":
    main()
