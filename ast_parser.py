"""
Utility module for accessing the Lean AST daemon.

Maintains a single persistent `dump_ast_server` process for the lifetime of
the Python interpreter.  Any module that imports `get_lean_ast` shares the
same process and the same lock — no matter how many files import this module.
"""

import atexit
import json
import os
import subprocess
import tempfile
import threading

REPL_DIR = os.environ.get("LEAN_WORKSPACE", os.path.join(os.getcwd(), "repl/"))


class SingletonASTDaemon:
    def __init__(self, repl_dir: str):
        self.repl_dir = repl_dir
        self.lock = threading.Lock()
        self.proc = None
        self._start_process()
        atexit.register(self.close)

    def _start_process(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

        self.proc = subprocess.Popen(
            ["lake", "exe", "dump_ast_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.repl_dir,
            bufsize=1,
        )

        # Warmup: preload Mathlib so the first real call is fast
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.repl_dir, delete=False, encoding="utf-8"
        ) as tf:
            tf.write("import Mathlib")
            tmp = tf.name
        try:
            print("[Server] New imports detected. Loading Environment...")
            self._get_ast_raw(tmp)
            print("[Server] Mathlib loaded. AST Daemon ready.")
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
        """Thread-safe: acquire lock, auto-restart if daemon crashed."""
        with self.lock:
            if self.proc.poll() is not None:
                print("[!] AST Daemon ngỏm, đang tự động hồi sinh...")
                self._start_process()
            return self._get_ast_raw(file_path)

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()


_SHARED_DAEMON = SingletonASTDaemon(REPL_DIR)


def get_lean_ast(code: str) -> list:
    """
    Return a list of AST block dicts for a Lean code string.
    Thread-safe; backed by a single persistent daemon process.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=REPL_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        return _SHARED_DAEMON.get_ast(path)
    finally:
        os.remove(path)
