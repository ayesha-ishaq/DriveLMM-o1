import os
import json

# Define input and output file paths
input_file = "DriveLMMo1_TEST.json"  # Change this to your actual path
output_file = "DriveLMMo1_TEST.jsonl"  # Change this to your actual path
# questions_file = "/data2/DriveLM/data/nuscenes/valset_1/valset_dataset.json"

# Read the combined JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
data = list(data.values())

# Open the output JSONL file for writing
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data:
        question = item['question']
        # stitched multiview image stored with idx as name
        filename = item['idx'].rsplit('_',1)[0] + '.png'
        jsonl_entry = {
            "id": item["idx"],
            "image": filename,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": f"{item['final_answer']}"
                }
            ]
        }
        # Write as a JSON line
        f_out.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

print(f"JSONL file written to {output_file}")
