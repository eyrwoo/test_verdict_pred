PRED_PROMPT_TMPL = '''
You are a skilled Python programmer and code reviewer.
Carefully evaluate if the given code passes all provided public and hidden test cases.

[You are given]
- Python code
- Test cases

[Your task]
- Determine whether the given Python code produces the correct result for the provided test input.

Requirements:
- Wrap the entire response in a single ```plaintext ... ``` code block.
- In [Result], only generate either PASS or FAIL (no other expressions allowed).

Output format:
EXAMPLE:
[Results]
```plaintext
[PASS/FAIL]
```
EXAMPLE_END:

Code:
```python
{code}
```

Testcase:
```test
{testcase}
```
'''

BUG_REPORT_PROMPT_TMPL = '''
You are a skilled Python programmer and code reviewer.
Carefully evaluate whether the given code can pass the given single test case, and present your reasoning and final judgment.

[You are given]
- Python code
- ONE test case (input, output, and optional function name)

[Your task]
- Analyze the functionality of the code and the provided test case,
  and determine whether the code will pass this specific test case.

[Requirements]
- Wrap the entire response in a ```plaintext ... ``` code block.
- In [Explanation], provide only the essential reasoning.
- In [Result], only generate either PASS or FAIL (no other expressions allowed).
- Do not modify the code or propose any new solution; perform only "review and judgment."

Output format
EXAMPLE:
[Explanation]
...
[Result]
```plaintext
[PASS/FAIL]
```
EXAMPLE_END:

Code:
```python
{code}
```

Testcase:
```test
{testcase}
```
'''

BUG_LOCAL_PROMPT_TMPL = '''
You are a skilled Python programmer and code reviewer.
Carefully evaluate whether the given code can pass all provided public and hidden test cases, and present your reasoning and final judgment.

[You are given]
- Python code
- Test cases

[Your task]
- Determine whether the given Python code produces the correct result for the provided test input.
- If the test fails, you must also identify the bug location in the code and briefly explain where the logic breaks.

[Requirements]
- Wrap the entire response in a ```plaintext ... ``` code block.
- In [Result], only generate either PASS or FAIL (no other expressions allowed).
- In [Bug localization], write only the bug location and explanation of the bug.
- Do not modify the code or propose any new solution; perform only "review and judgment."

Output format:
EXAMPLE:
[Result]
```plaintext
[PASS/FAIL]
```

[Bug Localization]
...
EXAMPLE_END:

Code:
```python
{code}
```

Testcase:
```test
{testcase}
```
'''
