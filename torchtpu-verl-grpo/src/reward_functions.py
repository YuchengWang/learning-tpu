"""
Rule-Based Reward Functions for GRPO Post-Training on Google Research MBPP
Includes Format Verification and Sandboxed Python Code Execution Rewards
"""

import re
import ast
import multiprocessing
from typing import List, Dict, Any

def extract_code_solution(text: str) -> str:
    """
    Extracts executable Python code from an <answer>...</answer> block.
    Extracts code within ```python ... ``` if markdown fences are present.
    """
    # 1. First extract inside <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    candidate_text = answer_match.group(1).strip() if answer_match else text

    # 2. Extract from markdown code blocks
    code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", candidate_text, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()

    # 3. Fallback: clean out tags if raw code is provided
    cleaned = re.sub(r"</?think>|</?answer>", "", candidate_text).strip()
    return cleaned

def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Evaluates adherence to structural reasoning format:
    <think>
    reasoning / plan / edge cases
    </think>
    <answer>
    ```python
    def solution(...):
        ...
    ```
    </answer>
    """
    rewards = []
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    for text in completions:
        text = text.strip()
        score = 0.0

        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        has_python_block = "```python" in text or "```" in text

        if has_think:
            score += 0.2
        if has_answer:
            score += 0.2
        if has_python_block:
            score += 0.1

        # Full pattern match bonus
        if re.match(pattern, text, re.DOTALL | re.IGNORECASE) and has_python_block:
            score = 0.5

        rewards.append(score)
    return rewards

def _run_single_test(code_str: str, test_list: List[str], test_setup_code: str, return_dict: dict):
    """Worker function executed inside separate process with restricted scope."""
    try:
        scope = {}
        if test_setup_code:
            exec(test_setup_code, scope)
        exec(code_str, scope)
        for assertion in test_list:
            exec(assertion, scope)
        return_dict["passed"] = True
    except Exception:
        return_dict["passed"] = False

def execute_code_with_timeout(code_str: str, test_list: List[str], test_setup_code: str = "", timeout: float = 2.0) -> bool:
    """Executes code against test assertions with a strict timeout to prevent hangs."""
    if not code_str:
        return False

    manager = multiprocessing.Manager()
    return_dict = manager.dict({"passed": False})

    p = multiprocessing.Process(
        target=_run_single_test,
        args=(code_str, test_list, test_setup_code, return_dict)
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    return return_dict.get("passed", False)

def code_execution_reward_func(completions: List[str], reward_models: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Executes extracted code against Google Research MBPP unit tests.
    Returns 1.0 if all test assertions pass, 0.0 otherwise.
    """
    rewards = []
    for pred_text, rm_info in zip(completions, reward_models):
        extracted_code = extract_code_solution(pred_text)
        test_list = rm_info.get("test_list", [])
        test_setup = rm_info.get("test_setup_code", "")

        passed = execute_code_with_timeout(extracted_code, test_list, test_setup, timeout=2.0)
        rewards.append(1.0 if passed else 0.0)

    return rewards

def compute_combined_mbpp_reward(completions: List[str], reward_models: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Combined Reward = Format Reward (0.0 to 0.5) + Unit Test Execution Reward (0.0 or 1.0)
    Total reward ranges from 0.0 to 1.5.
    """
    f_rewards = format_reward_func(completions, **kwargs)
    c_rewards = code_execution_reward_func(completions, reward_models, **kwargs)
    return [f + c for f, c in zip(f_rewards, c_rewards)]

def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs
) -> float:
    """
    VeRL standard custom reward entrypoint.
    Called by veRL NaiveRewardManager / BatchRewardManager during rollout evaluation.
    """
    if solution_str is None:
        return 0.0

    # 1. Format score (0.0 to 0.5)
    f_score = format_reward_func([solution_str])[0]

    # 2. Extract test assertions
    test_list = []
    test_setup_code = ""

    if extra_info and isinstance(extra_info, dict):
        test_list = extra_info.get("test_list", [])
        test_setup_code = extra_info.get("test_setup_code", "")

    if not test_list and ground_truth:
        if isinstance(ground_truth, str):
            try:
                import json
                gt_data = json.loads(ground_truth)
                if isinstance(gt_data, dict):
                    test_list = gt_data.get("test_list", [])
                    test_setup_code = gt_data.get("test_setup_code", "")
            except Exception:
                pass
        elif isinstance(ground_truth, dict):
            test_list = ground_truth.get("test_list", [])
            test_setup_code = ground_truth.get("test_setup_code", "")

    # 3. Sandboxed execution score (0.0 or 1.0)
    exec_score = 0.0
    if test_list:
        code = extract_code_solution(solution_str)
        passed = execute_code_with_timeout(code, test_list, test_setup_code, timeout=2.0)
        if passed:
            exec_score = 1.0

    return float(f_score + exec_score)

