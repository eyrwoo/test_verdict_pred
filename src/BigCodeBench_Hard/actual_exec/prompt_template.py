GENERATION_PROMPT_TMPL = """
You are an expert Python programmer.
You are given a "Code Skeleton" which includes imports, function signature, docstring, and potentially some implementation.
Your task is to implement the solution by completing the provided skeleton.

[Requirements]
- Your response MUST be a valid, self-contained Python program.
- Wrap the code in a single ```python``` code block.
- You MUST start your code with the EXACT content provided in the Code Skeleton (imports, function signature, default parameters, docstring).
- Do NOT modify the function signature or imports provided in the skeleton.
- Implement the function body to satisfy the requirements described in the docstring.

Output format:
[Full Code including imports, signature, docstring, and implementation]
```python
...
```

Code Skeleton:
```python
{prompt}
```
"""