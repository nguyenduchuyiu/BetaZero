"""Persistent Lean EXPR Tree daemon. Single shared process across all imports."""

import atexit
import json
import os
import resource
import signal
import subprocess
import tempfile
import threading

REPL_DIR = os.environ.get("LEAN_WORKSPACE", os.path.join(os.getcwd(), "repl/"))


class EXPRTreeDaemon:
    def __init__(self, repl_dir: str = REPL_DIR, max_requests: int = 500):
        self.repl_dir = repl_dir
        self.max_requests = max_requests
        self.request_count = 0
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

        def _preexec():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_AS, (1024**3*10, 1024**3*10))
        self.proc = subprocess.Popen(
            ["lake", "exe", "dump_expr_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.repl_dir,
            bufsize=1,
            preexec_fn=_preexec,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.repl_dir, delete=False, encoding="utf-8"
        ) as tf:
            tf.write('import Mathlib')
            tmp = tf.name
        try:
            print("[EXPR] Loading Mathlib environment...")
            self._get_expr_raw(tmp)
            print("[EXPR] Ready.")
        finally:
            os.remove(tmp)

    def _get_expr_raw(self, file_path: str) -> list:
        if not self.proc or not self.proc.stdin:
            return []
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

    def get_expr_tree(self, file_path: str) -> list:
        with self.lock:
            self.request_count += 1
            if (
                self.proc is None
                or self.proc.poll() is not None
                or self.request_count >= self.max_requests
            ):
                if self.request_count >= self.max_requests:
                    print(
                        f"[EXPR] Reached {self.max_requests} requests. Respawning to flush RAM..."
                    )
                else:
                    print("[EXPR] Daemon crashed or missing. Restarting...")
                self._start_process()
            return self._get_expr_raw(file_path)

    def close(self):
        self._kill_proc()

_SHARED_TREE_DAEMON = EXPRTreeDaemon(REPL_DIR)

def get_lean_expr_tree(code: str) -> list:
    prefix = "import Mathlib\n"
    full_code = code if "import " in code else prefix + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=REPL_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        path = f.name
        
    try:
        return _SHARED_TREE_DAEMON.get_expr_tree(path)
    finally:
        os.remove(path)