import json
import ast
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset

DEFAULT_DATASET_ID = "bigcode/bigcodebench-hard"
DEFAULT_SPLIT = "v0.1.4"

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSON or JSONL file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        if not text.strip():
            return []

    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    logging.warning("Skip invalid JSONL line: %s (%s)", line[:80], exc)
        return rows

    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return list(content.values())
    return []

def load_generated_codes(file_path):
    """
    Loads the generated codes from the specified JSON file.
    Expects the format found in 'nucleus_code_generate.json'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_bigcodebench_hard(dataset_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load BigCodeBench-Hard style problems.
    adapted from actual_exec/utils.py but preserves 'test' field for evaluation.
    """
    records: List[Dict[str, Any]] = []

    if dataset_path and os.path.exists(dataset_path):
        records = read_json_or_jsonl(dataset_path)
    else:
        try:
            ds = load_dataset(dataset_path or DEFAULT_DATASET_ID, split=DEFAULT_SPLIT)
            records = list(ds)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset from {dataset_path or DEFAULT_DATASET_ID} (split={DEFAULT_SPLIT}): {exc}"
            )

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(records):
        # Normalize task_id
        task_id = (
            item.get("task_id")
            or item.get("question_id")
            or item.get("problem_id")
            or item.get("id")
            or f"BigCodeBench_{idx}"
        )
        
        # Ensure 'test' field exists - this is critical for evaluation
        test_code = item.get("test")
        if not test_code:
            logging.warning("Skip record without test code: %s", task_id)
            continue

        if task_id == "BigCodeBench/1006":
             # Fix duplicate test case name in BigCodeBench/1006
             # There are two `test_non_zip_content` methods. The first one should be `test_valid_zip_url`.
             test_code = test_code.replace("def test_non_zip_content(self, mock_get):", "def test_valid_zip_url(self, mock_get):", 1)
            
        normalized.append({
            "task_id": str(task_id),
            "test": str(test_code)
            # We don't strictly need 'prompt' here for evaluation, but could keep it if needed
        })
            
    return normalized

def split_test_cases(test_case_code):
    """
    Splits a unittest class string into individual test cases.
    Each returned string contains:
    - Global imports and statements
    - The Class definition
    - setUp/tearDown methods (if present)
    - One specific test_ method
    Returns:
    List[Tuple[str, str]]: A list of (test_method_name, full_test_case_code) tuples
    """
    try:
        tree = ast.parse(test_case_code)
    except SyntaxError:
        # Fallback using a dummy name if parsing fails
        return [("error_parsing", test_case_code)]

    # Separating imports/global level stuff from the Class
    imports_and_globals = []
    test_class_node = None
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # We assume there is only one main test class usually named TestCases or similar
            # But we'll take the first class we find or specifically look for one inheriting from unittest.TestCase if strictness needed
            # For BigCodeBench, usually it is `class TestCases(unittest.TestCase):`
            test_class_node = node
        else:
            imports_and_globals.append(node)
            

    if not test_class_node:
        return [("no_class_found", test_case_code)]
        
    # Extract setUp, tearDown, and test_ methods
    setup_method = None
    teardown_method = None
    test_methods = []
    other_methods = [] # Helper methods inside the class
    
    for item in test_class_node.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == 'setUp':
                setup_method = item
            elif item.name == 'tearDown':
                teardown_method = item
            elif item.name.startswith('test'):
                test_methods.append(item)
            else:
                other_methods.append(item)
        else:
             # Class docstrings or assignments
            other_methods.append(item)
            

    if not test_methods:
         return [("no_test_methods", test_case_code)]
         
    split_cases = []
    
    # Reconstruct for each test method
    for test_method in test_methods:
        # New class body
        new_body = []
        new_body.extend(other_methods)
        if setup_method:
            new_body.append(setup_method)
        if teardown_method:
            new_body.append(teardown_method)
        new_body.append(test_method)
        
        # Sort body by lineno to keep somewhat original order if possible, 
        # but pure reconstruction is fine too. 
        # Actually usually order doesn't matter for execution unless dependencies exist.
        # But `other_methods` might be helpers used by tests.
        
        # Create new ClassDef
        new_class = ast.ClassDef(
            name=test_class_node.name,
            bases=test_class_node.bases,
            keywords=test_class_node.keywords,
            body=new_body,
            decorator_list=test_class_node.decorator_list
        )
        
        # Create new module
        new_module_body = imports_and_globals + [new_class]
        new_module = ast.Module(body=new_module_body, type_ignores=[])
        
        # Unparse to string
        split_cases.append((test_method.name, ast.unparse(new_module)))
        
    return split_cases
