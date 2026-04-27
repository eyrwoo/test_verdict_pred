GENERATION_PROMPT_TMPL = """You are an expert competitive programmer.
You are given a programming problem. Implement a complete, correct solution.

[Requirements]
- Your response MUST contain exactly one ```python``` code block with a complete, runnable solution.
- Read input from stdin and write output to stdout.
- If a starter code template is provided, your solution MUST start with the exact content of the starter code.
- Do NOT include any explanation outside the code block.

Output format:
```python
...
```

Problem:
{question_content}
{starter_code_section}"""

STARTER_CODE_SECTION_TMPL = """
Starter code (your solution MUST begin with this):
```python
{starter_code}
```"""
