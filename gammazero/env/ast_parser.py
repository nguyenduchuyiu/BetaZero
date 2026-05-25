"""Persistent Lean AST daemon. Single shared process across all imports."""

import atexit
import json
import os
import signal
import subprocess
import tempfile
import threading

REPL_DIR = os.environ.get("LEAN_WORKSPACE", os.path.join(os.getcwd(), "repl/"))


class ASTDaemon:
    def __init__(self, repl_dir: str = REPL_DIR, max_requests: int = 2000):
        self.repl_dir = repl_dir
        self.max_requests = max_requests
        self.lock = threading.Lock()
        self.proc = None
        self._start_process()
        atexit.register(self.close)

    def _kill_proc(self):
        proc_to_kill = self.proc
        self.proc = None
        if proc_to_kill:
            try:
                if proc_to_kill.poll() is None:
                    os.killpg(os.getpgid(proc_to_kill.pid), signal.SIGKILL)
                    proc_to_kill.wait(timeout=2)
            except Exception:
                pass

    def _start_process(self):
        self._kill_proc()
        self.request_count = 0
        
        self.proc = subprocess.Popen(
            ["lake", "exe", "dump_ast_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.repl_dir,
            bufsize=1,
            preexec_fn=os.setsid  
        )
        
        # Warmup: cache the Mathlib environment so the first real call is fast.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.repl_dir, delete=False, encoding="utf-8"
        ) as tf:
            tf.write('import Mathlib')
            tmp = tf.name
        try:
            print("[AST] Loading Mathlib environment...")
            self._get_ast_raw(tmp)
            print("[AST] Ready.")
        finally:
            os.remove(tmp)

    def _get_ast_raw(self, file_path: str) -> list:
        self.proc.stdin.write(file_path + "\n")
        self.proc.stdin.flush()
        blocks = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "===EOF===":
                break
            if line.startswith("{"):
                try:
                    blocks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return blocks

    def get_ast(self, file_path: str) -> list:
        with self.lock:
            self.request_count += 1
            
            if self.proc is None or self.proc.poll() is not None or self.request_count >= self.max_requests:
                if self.request_count >= self.max_requests:
                    print(f"\n[AST Server] Reached {self.max_requests} requests. Respawning to flush memory...")
                self._start_process()

            try:
                return self._get_ast_raw(file_path)
            except Exception as e:
                print(f"[AST Server] Error during AST dump: {e}. Respawning...")
                self._start_process()  
                return []

    def close(self):
        self._kill_proc()


_SHARED_DAEMON = ASTDaemon()


def get_lean_ast(code: str) -> list:
    """Return AST block dicts for a Lean code string.

    Thread-safe; auto-injects `import Mathlib` if missing and corrects byte
    offsets for the injected prefix.
    """

    prefix = "import Mathlib\n"
    has_import = "import " in code

    if not has_import:
        full_code = prefix + code
        offset_bytes = len(prefix.encode('utf-8'))  # 15 bytes
    else:
        full_code = code
        offset_bytes = 0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=REPL_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        path = f.name

    try:
        blocks = _SHARED_DAEMON.get_ast(path)

        if offset_bytes > 0:
            valid_blocks = []
            for b in blocks:
                b["start_byte"] -= offset_bytes
                b["end_byte"] -= offset_bytes

                # Drop nodes from the synthetic import line; keep real ones (>= 0).
                if b["start_byte"] >= 0:
                    valid_blocks.append(b)
            return valid_blocks

        return blocks
    finally:
        os.remove(path)
