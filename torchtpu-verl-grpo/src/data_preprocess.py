"""
Data Preprocessing for VeRL GRPO Training on Cloud TPU
Processes Google Research MBPP (Mostly Basic Python Problems) into VeRL Parquet Schema
Official Source: https://github.com/google-research/google-research/tree/master/mbpp
"""

import os
import json
import argparse
import urllib.request
import pandas as pd

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a programming problem in Python, "
    "and the Assistant provides the solution.\n"
    "The assistant first thinks about the problem decomposition, algorithms, and potential edge cases "
    "inside <think> </think> tags.\n"
    "Then, the assistant provides the complete, working Python implementation wrapped in an "
    "<answer> ```python ... ``` </answer> block."
)

MBPP_RAW_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

def download_mbpp_data(url: str = MBPP_RAW_URL):
    """Downloads the official Google Research MBPP dataset directly via HTTP."""
    print(f"🌐 Fetching official Google Research MBPP dataset from: {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AI-GDE-TorchTPU-Sprint)"})
    records = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str:
                records.append(json.loads(line_str))
    print(f"✅ Successfully downloaded {len(records)} MBPP task records.")
    return records

def format_prompt(problem_text: str, test_list: list) -> list:
    """Formats the user prompt including the problem description and sample assertion."""
    example_test = test_list[0] if test_list else ""
    user_query = f"{problem_text}\n\nYour code should satisfy the following test assertion:\n```python\n{example_test}\n```"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

def process_mbpp(output_dir: str = "data/mbpp", max_train: int = None, max_test: int = None):
    os.makedirs(output_dir, exist_ok=True)
    raw_records = download_mbpp_data()

    # Standard Google Research MBPP partition:
    # Tasks 11 - 510: Training set
    # Tasks 511 - 974: Evaluation / Test set
    train_records = [r for r in raw_records if 11 <= r["task_id"] <= 510]
    test_records = [r for r in raw_records if r["task_id"] > 510]

    if max_train:
        train_records = train_records[:max_train]
    if max_test:
        test_records = test_records[:max_test]

    train_data = []
    for r in train_records:
        test_list = r.get("test_list", [])
        test_setup = r.get("test_setup_code", "")
        gt_code = r.get("code", "")
        train_data.append({
            "data_source": "google_mbpp",
            "prompt": format_prompt(r["text"], test_list),
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"test_list": test_list, "test_setup_code": test_setup, "code": gt_code}),
                "task_id": r["task_id"],
            },
            "extra_info": {
                "task_id": r["task_id"],
                "split": "train",
                "test_list": test_list,
                "test_setup_code": test_setup,
                "ground_truth": gt_code,
            }
        })

    test_data = []
    for r in test_records:
        test_list = r.get("test_list", [])
        test_setup = r.get("test_setup_code", "")
        gt_code = r.get("code", "")
        test_data.append({
            "data_source": "google_mbpp",
            "prompt": format_prompt(r["text"], test_list),
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"test_list": test_list, "test_setup_code": test_setup, "code": gt_code}),
                "task_id": r["task_id"],
            },
            "extra_info": {
                "task_id": r["task_id"],
                "split": "test",
                "test_list": test_list,
                "test_setup_code": test_setup,
                "ground_truth": gt_code,
            }
        })

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"📊 Dataset Stats:")
    print(f"  - Training samples: {len(train_df)} -> {train_path}")
    print(f"  - Testing samples:  {len(test_df)} -> {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/mbpp", help="Output directory for MBPP Parquet files")
    parser.add_argument("--max_train", type=int, default=None, help="Optional max train samples")
    parser.add_argument("--max_test", type=int, default=None, help="Optional max test samples")
    args = parser.parse_args()

    process_mbpp(args.output_dir, args.max_train, args.max_test)
