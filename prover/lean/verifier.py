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

def verify_lean4_file(code, timeout=300, **kwargs):
    """Hàm verify lõi, nhận code dạng string và trả về dict kết quả."""
    command = {"cmd": code}
    message_str = json.dumps(command, ensure_ascii=False)
    start_time = time.time()
    
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            
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
                # Kill the whole process group so spawned `repl` is terminated too.
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
        result = {"pass": False, "complete": False, "system_errors": "Timeout", "errors": []}
    except json.JSONDecodeError:
        result = {"pass": False, "complete": False, "system_errors": f"JSON Error. Stdout: {stdout}", "errors": []}
    except Exception as e:
        result = {"pass": False, "complete": False, "system_errors": str(e), "errors": []}
        
    result['verify_time'] = time.time() - start_time
    return result


class Lean4ServerScheduler:
    """
    Scheduler persistent: Duy trì dàn worker liên tục để nhận task mới.
    Tương thích hoàn toàn với interface của DeepSeek-Prover.
    """
    def __init__(self, max_concurrent_requests=1, timeout=300, name='verifier', **kwargs):
        self.timeout = timeout
        # Khởi tạo Pool 1 lần, dàn worker sẽ túc trực ở đây cho đến khi gọi .close()
        self.executor = ProcessPoolExecutor(max_workers=max_concurrent_requests)
        self.futures = {} # Lưu trạng thái các task đang chạy
        print(f"[{name}] Đã khởi động scheduler với {max_concurrent_requests} worker(s).")

    def submit_all_request(self, tasks):
        """
        Nhận 1 list các dictionary (vd: [{'code': '...'}])
        Ném vào cho worker xử lý và trả về list chứa các ID của request.
        """
        request_ids = []
        for task in tasks:
            req_id = str(uuid.uuid4()) # Tạo ID ngẫu nhiên cho mỗi task
            code = task.get('code', '')
            task_timeout = task.get('timeout', self.timeout)
            
            # Giao việc cho worker, lưu lại cái 'thẻ hẹn' (future)
            future = self.executor.submit(verify_lean4_file, code, task_timeout)
            self.futures[req_id] = future
            request_ids.append(req_id)
            
        return request_ids

    def get_all_request_outputs(self, request_ids):
        """
        Chờ và thu thập kết quả từ các ID tương ứng.
        """
        outputs = []
        for req_id in request_ids:
            if req_id in self.futures:
                # Hàm .result() sẽ chặn (block) cho đến khi file đó chạy xong
                result = self.futures[req_id].result()
                outputs.append(result)
                # Lấy xong kết quả thì xoá thẻ hẹn đi cho nhẹ RAM
                del self.futures[req_id] 
                
        return outputs

    def close(self):
        """
        Dọn dẹp và đóng dàn worker. Bắt buộc gọi ở cuối script.
        """
        self.executor.shutdown(wait=True)
        print("Đã đóng Lean4ServerScheduler an toàn.")
