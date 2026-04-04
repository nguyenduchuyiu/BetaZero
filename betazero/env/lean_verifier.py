import os
import time
import json
import subprocess
import shutil
import uuid
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

DEFAULT_LAKE_PATH = shutil.which("lake") or "lake"
DEFAULT_LEAN_WORKSPACE = os.path.join(os.getcwd(), "repl/")


import os
import signal # Thêm cái này
import time
import json
import subprocess
import shutil
import uuid
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

DEFAULT_LAKE_PATH = shutil.which("lake") or "lake"
DEFAULT_LEAN_WORKSPACE = os.path.join(os.getcwd(), "repl/")

class PersistentLeanWorker:
    """A persistent Lean REPL process that caches Mathlib in its base environment."""
    
    def __init__(self, workspace=DEFAULT_LEAN_WORKSPACE, timeout=20, max_requests=500): 
        self.workspace = workspace
        self.timeout = timeout
        self.max_requests = max_requests 
        self.request_count = 0           
        self.proc = None
        self.base_env = None
        self.lock = threading.Lock()
        self._is_timed_out = False # Cờ báo hiệu bom nổ
        self._start_repl()

    def _kill_proc(self):
        # Lấy proc ra và gán None ngay lập tức để tránh Race Condition với Thread bom
        proc_to_kill = self.proc
        self.proc = None
        if proc_to_kill:
            try:
                if proc_to_kill.poll() is None:
                    os.killpg(os.getpgid(proc_to_kill.pid), signal.SIGKILL)
                    proc_to_kill.wait(timeout=2)
            except Exception as e:
                pass # Nuốt lỗi vì process có thể đã chết

    def _timeout_handler(self):
        """Hàm kích nổ: Gây án mạng khi hết giờ."""
        self._is_timed_out = True
        self._kill_proc()

    def _read_json_response(self):
        buffer = ""
        while True:
            # Nếu _kill_proc() được gọi từ Timer, proc sẽ bị None hoặc pipe bị đứt
            if not self.proc or not self.proc.stdout:
                raise RuntimeError("Process killed by timeout or died.")
                
            line = self.proc.stdout.readline()
            
            if not line:
                if buffer:
                    print(f"\n[CRITICAL FATAL] Lean REPL đột tử khi đang in JSON! Code dở dang:\n{buffer}")
                raise RuntimeError("REPL closed unexpectedly (EOF). Lean Process died.")
            
            if not buffer and not line.strip().startswith("{"):
                print(f"[REPL GARBAGE] {line.strip()}")
                continue
            
            buffer += line
            
            try:
                parsed_json = json.loads(buffer)
                return parsed_json
            except json.JSONDecodeError:
                continue

    def _start_repl(self):
        self._kill_proc()
        self.request_count = 0
        self._is_timed_out = False
        
        self.proc = subprocess.Popen(
            [DEFAULT_LAKE_PATH, "exe", "repl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.workspace,
            bufsize=1,
            preexec_fn=os.setsid
        )
        warmup_cmd = json.dumps({"cmd": "import Mathlib"})
        
        self.proc.stdin.write(warmup_cmd + "\n\n")
        self.proc.stdin.flush()
        
        try:
            res_json = self._read_json_response()
            self.base_env = res_json.get("env")
            print(f"[Worker PID {self.proc.pid}] Mathlib warmed up. Base env: {self.base_env}")
        except Exception as e:
            print(f"[Worker PID {self.proc.pid if self.proc else 'N/A'}] Warmup failed: {e}")
            self.base_env = None

    def verify(self, code: str) -> dict:
        start_time = time.time()
        
        with self.lock:
            self.request_count += 1 
            
            if self.proc is None or self.proc.poll() is not None or self.request_count >= self.max_requests:
                if self.request_count >= self.max_requests:
                    print(f"Worker reached {self.max_requests} requests. Respawning to flush RAM leak...")
                else:
                    print("Worker died or missing. Respawning...")
                self._start_repl()

            payload = {"cmd": code}
            if self.base_env is not None:
                payload["env"] = self.base_env

            message_str = json.dumps(payload, ensure_ascii=False)
            
            # GÀI BOM HẸN GIỜ!
            self._is_timed_out = False
            timer = threading.Timer(self.timeout, self._timeout_handler)
            timer.start()
            
            try:
                self.proc.stdin.write(message_str + "\n\n")
                self.proc.stdin.flush()
                
                result_json = self._read_json_response()
                
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
                    "system_errors": "",
                    "verified_code": code,
                }
            except Exception as e:
                # Kiểm tra xem có phải do bom nổ không
                if self._is_timed_out:
                    sys_err = f"Lean verification TIMED OUT after {self.timeout} seconds!"
                else:
                    sys_err = str(e)
                    
                result = {"pass": False, "complete": False, "system_errors": sys_err, "errors": []}
                print(f"[Worker] Failed/Timeout: {sys_err}. Respawning...")
                self._start_repl() # Thay máu công nhân mới
            finally:
                timer.cancel() # Gỡ bom nếu Lean chạy xong sớm!

        result["verify_time"] = time.time() - start_time
        return result

    def close(self):
        self._kill_proc()


class Lean4ServerScheduler:
    """Thread pool managing persistent Stateful Lean Workers."""

    def __init__(self, max_concurrent_requests=1, timeout=60, name="verifier", **kwargs):
        self.timeout = timeout
        self.worker_queue = Queue()
        self.workers = []
        
        print(f"[{name}] Booting {max_concurrent_requests} persistent Lean worker(s)... (This takes ~5 seconds)")
        for _ in range(max_concurrent_requests):
            worker = PersistentLeanWorker(timeout=self.timeout)
            self.workers.append(worker)
            self.worker_queue.put(worker)
            
        # Dùng ThreadPool thay vì ProcessPool vì các Worker object có chứa Popen subprocess
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.futures: dict = {}
        print(f"[{name}] Scheduler Ready.")

    def _verify_task(self, code: str):
        worker = self.worker_queue.get()
        try:
            res = worker.verify(code)
            return res
        finally:
            self.worker_queue.put(worker)

    def submit_all_request(self, tasks) -> list[str]:
        request_ids = []
        for task in tasks:
            req_id = str(uuid.uuid4())
            self.futures[req_id] = self.executor.submit(
                self._verify_task, task.get("code", "")
            )
            request_ids.append(req_id)
        return request_ids

    def get_all_request_outputs(self, request_ids) -> list[dict]:
        return [self.futures.pop(req_id).result() for req_id in request_ids if req_id in self.futures]

    def verify(self, code: str) -> dict:
        req_id = self.submit_all_request([{"code": code}])[0]
        return self.get_all_request_outputs([req_id])[0]

    def close(self):
        self.executor.shutdown(wait=True)
        for worker in self.workers:
            worker.close()
        print("Lean4ServerScheduler closed and all REPLs terminated.")