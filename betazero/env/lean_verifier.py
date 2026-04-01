import os
import time
import json
import subprocess
import tempfile
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
import signal

DEFAULT_LAKE_PATH = shutil.which("lake") or "lake"
DEFAULT_LEAN_WORKSPACE = os.path.join(os.getcwd(), "repl/")


def verify_lean_code(code, timeout=300):
    """Verify Lean code via the REPL; return a result dict."""
    message_str = json.dumps({"cmd": code}, ensure_ascii=False)
    start_time = time.time()
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as tmp:
            tmp.write(message_str + "\r\n\r\n")
            tmp.seek(0)
            proc = subprocess.Popen(
                [DEFAULT_LAKE_PATH, "exe", "repl"],
                stdin=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=DEFAULT_LEAN_WORKSPACE,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the entire process group to terminate the spawned repl
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise
        result_json = json.loads(stdout)
        messages = result_json.get("messages", [])
        errors   = [m for m in messages if m.get("severity") == "error"]
        warnings = [m for m in messages if m.get("severity") == "warning"]
        sorries  = result_json.get("sorries", [])
        is_pass  = len(errors) == 0
        has_sorry_warning = any(
            "declaration uses 'sorry'" in w.get("data", "") or "failed" in w.get("data", "")
            for w in warnings
        )
        result = {
            "pass": is_pass,
            "complete": is_pass and not sorries and not has_sorry_warning,
            "errors": errors,
            "warnings": warnings,
            "infos": [m for m in messages if m.get("severity") == "info"],
            "sorries": sorries,
            "system_errors": stderr or "",
            "verified_code": code,
        }
    except subprocess.TimeoutExpired:
        result = {"pass": False, "complete": False, "system_errors": "Timeout", "errors": []}
    except json.JSONDecodeError:
        result = {"pass": False, "complete": False, "system_errors": f"JSON Error. Stdout: {stdout}", "errors": []}
    except Exception as e:
        result = {"pass": False, "complete": False, "system_errors": str(e), "errors": []}
    result["verify_time"] = time.time() - start_time
    return result


class Lean4ServerScheduler:
    """Process pool for concurrent Lean verification. Compatible with DeepSeek-Prover interface."""

    def __init__(self, max_concurrent_requests=1, timeout=300, name="verifier", **kwargs):
        self.timeout = timeout
        self.executor = ProcessPoolExecutor(max_workers=max_concurrent_requests)
        self.futures: dict = {}
        print(f"[{name}] Scheduler started with {max_concurrent_requests} worker(s).")

    def submit_all_request(self, tasks) -> list[str]:
        """Submit a batch of {'code', 'timeout'} tasks; return request IDs."""
        request_ids = []
        for task in tasks:
            req_id = str(uuid.uuid4())
            self.futures[req_id] = self.executor.submit(
                verify_lean_code, task.get("code", ""), task.get("timeout", self.timeout)
            )
            request_ids.append(req_id)
        return request_ids

    def get_all_request_outputs(self, request_ids) -> list[dict]:
        """Block until all given request IDs complete; return results."""
        return [self.futures.pop(req_id).result() for req_id in request_ids if req_id in self.futures]

    def verify(self, code: str) -> dict:
        """Synchronous single-code verification through the pool."""
        req_id = self.submit_all_request([{"code": code, "timeout": self.timeout}])[0]
        return self.get_all_request_outputs([req_id])[0]

    def close(self):
        """Shutdown worker pool. Call at program exit."""
        self.executor.shutdown(wait=True)
        print("Lean4ServerScheduler closed.")
