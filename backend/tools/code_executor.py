import base64
from e2b_code_interpreter import Sandbox
from backend.config import E2B_API_KEY


def run_code_in_sandbox(code: str) -> dict:
    """
    Executes Python code in a secure E2B cloud sandbox.
    Returns stdout, stderr, and a base64-encoded chart image if one was saved.
    """
    with Sandbox(api_key=E2B_API_KEY) as sandbox:
        execution = sandbox.run_code(code)

        stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
        stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""
        chart_b64 = ""

        # Check if a chart was saved as chart.png inside the sandbox
        for result in execution.results:
            if result.png:
                chart_b64 = result.png  # already base64 encoded by E2B

        return {
            "stdout": stdout,
            "stderr": stderr,
            "chart_image": chart_b64,
            "error": execution.error.value if execution.error else None,
        }
