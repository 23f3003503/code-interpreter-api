
import os
import sys
import json
import tempfile
import subprocess
import traceback
import re

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Code Interpreter with AI Error Analysis")

# Enable browser-based testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: list[int]
    result: str


def execute_python_code(code: str) -> dict:
    """Execute Python code and capture its output or traceback."""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        completed = subprocess.run(
            [sys.executable, "-I", temp_path],
            capture_output=True,
            text=True,
            timeout=5
        )

        if completed.returncode == 0:
            return {
                "success": True,
                "output": completed.stdout
            }

        return {
            "success": False,
            "output": completed.stdout + completed.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "Execution timed out after 5 seconds."
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_error_with_ai(code: str, error_output: str) -> list[int]:
    """Ask AIPipe to identify erroneous lines in the submitted code."""
    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is missing.")

    prompt = f"""
Analyze the Python execution failure.

Identify the 1-based line number(s) in the submitted code
responsible for the error. Use the traceback as evidence.
Do not report lines belonging to Python libraries.
Do not invent line numbers. Return an empty list if no
submitted-code line can be established.

Return JSON only in this format:
{{"error_lines": [3]}}

SUBMITTED CODE:
{code}

EXECUTION TRACEBACK:
{error_output}
"""

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You analyze Python tracebacks. "
                        "Return only valid JSON with an error_lines "
                        "array of integer line numbers."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0
        },
        timeout=30
    )

    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    lines = parsed.get("error_lines", [])

    if not isinstance(lines, list):
        return []

    valid_lines = []
    for line in lines:
        if (
            isinstance(line, int)
            and not isinstance(line, bool)
            and 1 <= line <= len(code.splitlines())
        ):
            valid_lines.append(line)

    return sorted(set(valid_lines))


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI error analysis failed: {exc}"
        )

    return {
        "error": error_lines,
        "result": execution["output"]
    }


@app.get("/")
def health_check():
    return {"status": "ok", "endpoint": "/code-interpreter"}
