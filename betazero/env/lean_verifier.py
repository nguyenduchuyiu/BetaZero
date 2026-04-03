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


class PersistentLeanWorker:
    """A persistent Lean REPL process that caches Mathlib in its base environment."""
    
    def __init__(self, workspace=DEFAULT_LEAN_WORKSPACE, timeout=60):
        self.workspace = workspace
        self.timeout = timeout
        self.proc = None
        self.base_env = None
        self.lock = threading.Lock()
        self._start_repl()

    def _read_json_response(self):
        """Hút rác và tự động gom (accumulate) các dòng lại nếu Lean in JSON trên nhiều dòng."""
        buffer = ""
        while True:
            line = self.proc.stdout.readline()
            
            # Xử lý trường hợp Lean bị đột tử (OOM / Segfault) khi đang in dở JSON
            if not line:
                if buffer:
                    print(f"\n[CRITICAL FATAL] Lean REPL đột tử khi đang in JSON! Code dở dang:\n{buffer}")
                raise RuntimeError("REPL closed unexpectedly (EOF). Lean Process died.")
            
            # Lọc rác ở những dòng đầu tiên (trước khi dấu ngoặc '{' bắt đầu)
            if not buffer and not line.strip().startswith("{"):
                print(f"[REPL GARBAGE] {line.strip()}")
                continue
            
            # Nhét dòng mới đọc vào kho
            buffer += line
            
            try:
                # Thử parse toàn bộ kho hiện tại
                parsed_json = json.loads(buffer)
                return parsed_json  # Thành công! Thoát vòng lặp và trả về JSON chuẩn
            except json.JSONDecodeError:
                # Nếu văng lỗi -> Nghĩa là JSON chưa đóng ngoặc xong (JSON nhiều dòng)
                # Kệ mẹ nó, tiếp tục vòng lặp để lấy dòng tiếp theo ghép vào!
                continue

    def _start_repl(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        
        self.proc = subprocess.Popen(
            [DEFAULT_LAKE_PATH, "exe", "repl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.workspace,
            bufsize=1, # Line buffered
        )
        warmup_cmd = json.dumps({"cmd": "import Mathlib"})
        
        # CHỈ GỬI ĐÚNG 1 DẤU \n THÔI
        self.proc.stdin.write(warmup_cmd + "\n\n")
        self.proc.stdin.flush()
        
        try:
            res_json = self._read_json_response()
            self.base_env = res_json.get("env")
            print(f"[Worker PID {self.proc.pid}] Mathlib warmed up. Base env: {self.base_env}")
        except Exception as e:
            print(f"[Worker PID {self.proc.pid}] Warmup failed: {e}")
            self.base_env = None

    def verify(self, code: str) -> dict:
        """Verify code strictly branching from the warmed-up base_env."""
        start_time = time.time()
        
        with self.lock:
            # Nếu process chết (ví dụ rò rỉ RAM OOM), respawn lại
            if self.proc.poll() is not None:
                print("Worker died. Respawning...")
                self._start_repl()

            payload = {"cmd": code}
            if self.base_env is not None:
                payload["env"] = self.base_env

            message_str = json.dumps(payload, ensure_ascii=False)
            
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
                result = {"pass": False, "complete": False, "system_errors": str(e), "errors": []}
                # Khởi động lại nếu lỗi quá nặng
                self._start_repl()

        result["verify_time"] = time.time() - start_time
        return result

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()


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