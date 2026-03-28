import os
import time
import json
import subprocess
import tempfile
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
import signal

# Default path for the lake executable
DEFAULT_LAKE_PATH = shutil.which("lake") or "lake"
# Default Lean workspace directory
DEFAULT_LEAN_WORKSPACE = os.path.join(os.getcwd(), "repl/")

def verify_lean_code(code, timeout=300):
    """
    Core verification function. Receives Lean code as a string and returns a result dict.
    """
    command = {"cmd": code}
    message_str = json.dumps(command, ensure_ascii=False)
    start_time = time.time()
    
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            # Write the command to the temporary file and seek back to the start
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            
            # Start the Lean REPL process
            proc = subprocess.Popen(
                [DEFAULT_LAKE_PATH, "exe", 'repl'],
                stdin=temp_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=DEFAULT_LEAN_WORKSPACE,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the whole process group so the spawned `repl` is terminated as well
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise
        system_messages = stderr or ''
        result_json = json.loads(stdout)
        messages = result_json.get('messages', [])
        
        errors = [m for m in messages if m.get('severity') == 'error']
        warnings = [m for m in messages if m.get('severity') == 'warning']
        infos = [m for m in messages if m.get('severity') == 'info']
        sorries = result_json.get('sorries', [])
        
        is_pass = len(errors) == 0
        # Consider warnings about 'sorry' or 'failed' as incomplete
        has_sorry_or_failed_warning = any(
            "declaration uses 'sorry'" in w.get('data', '') or 'failed' in w.get('data', '') 
            for w in warnings
        )
        is_complete = is_pass and len(sorries) == 0 and not has_sorry_or_failed_warning
        
        result = {
            "pass": is_pass,
            "complete": is_complete,
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "sorries": sorries,
            "system_errors": system_messages,
            "verified_code": code
        }

    except subprocess.TimeoutExpired:
        # Timed out waiting for Lean
        result = {"pass": False, "complete": False, "system_errors": "Timeout", "errors": []}
    except json.JSONDecodeError:
        # The process output was not valid JSON
        result = {"pass": False, "complete": False, "system_errors": f"JSON Error. Stdout: {stdout}", "errors": []}
    except Exception as e:
        # Catch-all for any other exception
        result = {"pass": False, "complete": False, "system_errors": str(e), "errors": []}
        
    result['verify_time'] = time.time() - start_time
    return result


class Lean4ServerScheduler:
    """
    Persistent scheduler that maintains a pool of workers to process new tasks.
    Fully compatible with the DeepSeek-Prover interface.
    """
    def __init__(self, max_concurrent_requests=1, timeout=300, name='verifier', **kwargs):
        self.timeout = timeout
        # Initialize the process pool once; workers remain alive until .close() is called
        self.executor = ProcessPoolExecutor(max_workers=max_concurrent_requests)
        self.futures = {}  # Store futures representing running tasks
        print(f"[{name}] Scheduler started with {max_concurrent_requests} worker(s).")

    def submit_all_request(self, tasks):
        """
        Accept a list of task dictionaries (e.g., [{'code': '...'}])
        Submits each to the worker pool and returns a list of request IDs.
        """
        request_ids = []
        for task in tasks:
            req_id = str(uuid.uuid4())  # Generate a unique ID for each task
            code = task.get('code', '')
            task_timeout = task.get('timeout', self.timeout)
            
            # Submit the job to an executor worker and save the future
            future = self.executor.submit(verify_lean_code, code, task_timeout)
            self.futures[req_id] = future
            request_ids.append(req_id)
            
        return request_ids

    def get_all_request_outputs(self, request_ids):
        """
        Waits for and collects results for the specified request IDs.
        """
        outputs = []
        for req_id in request_ids:
            if req_id in self.futures:
                # .result() blocks until the worker process for this request finishes
                result = self.futures[req_id].result()
                outputs.append(result)
                # Remove the future to free up memory after result is collected
                del self.futures[req_id] 
                
        return outputs

    def close(self):
        """
        Clean up and close the worker pool. Must be called at the end of the script.
        """
        self.executor.shutdown(wait=True)
        print("Lean4ServerScheduler closed safely.")