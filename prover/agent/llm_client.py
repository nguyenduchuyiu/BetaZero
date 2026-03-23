from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - minimal venv
    httpx = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover
    _load_dotenv = None


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _offline_stub_enabled() -> bool:
    """Chỉ dùng cho dev/test: bật bằng LOCAL_LLM_MOCK=1, không gọi server."""
    v = _env("LOCAL_LLM_MOCK", "").lower()
    return v in ("1", "true", "yes", "on")


class LLMClient:
    """
    Mọi lệnh `complete()` gọi **một model local** qua HTTP API kiểu OpenAI Chat Completions
    (vLLM, llama.cpp server, LM Studio, Ollama `/v1`, text-generation-webui, …).
    Tên model phải trùng với id mà server local của bạn khai báo (`LOCAL_LLM_MODEL`).
    """

    def __init__(self) -> None:
        if _load_dotenv is not None:
            _load_dotenv()
        self.base_url = _env("LOCAL_LLM_BASE_URL").rstrip("/")
        self.api_key = _env("LOCAL_LLM_API_KEY")
        self.model = _env("LOCAL_LLM_MODEL")
        self.chat_path = _env("LOCAL_LLM_CHAT_PATH", "v1/chat/completions").strip("/")
        self.timeout = float(_env("LOCAL_LLM_TIMEOUT_SEC", "120") or "120")

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        task: str = "generic",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        if _offline_stub_enabled():
            return self._offline_stub_complete(prompt, system=system, task=task)

        if not self.base_url:
            raise RuntimeError(
                "Thiếu LOCAL_LLM_BASE_URL. "
                "Trỏ tới endpoint local (vd. http://127.0.0.1:1234/v1) hoặc bật LOCAL_LLM_MOCK=1 để stub offline."
            )
        if not self.model:
            raise RuntimeError(
                "Thiếu LOCAL_LLM_MODEL. "
                "Đặt đúng id model mà server local của bạn nhận (tên tùy server, ví dụ llama3.1, qwen2.5-coder, …)."
            )

        url = f"{self.base_url}/{self.chat_path}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if httpx is not None:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
        else:
            data = self._post_json_urllib(url, headers, payload)
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()

    def _offline_stub_complete(self, prompt: str, system: str, task: str) -> str:
        """Phản hồi giả khi LOCAL_LLM_MOCK=1 — không thay cho model local thật."""
        t = (task or "generic").lower()
        if t == "planner":
            stmt = prompt.strip()[:200]
            skel = (
                "import Mathlib\n\n"
                "theorem mock_goal : True := by\n"
                "  sorry\n"
            )
            return json.dumps(
                {
                    "informal_plan": f"[STUB] High-level steps to prove: {stmt!r} ...",
                    "skeleton_code": skel,
                },
                ensure_ascii=False,
            )
        if t == "router":
            return "SOLVE"
        if t == "decompose":
            return json.dumps(
                {
                    "subgoals": [
                        "subgoal_a : True",
                        "subgoal_b : True",
                    ]
                },
                ensure_ascii=False,
            )
        if t == "solver":
            return (
                "import Mathlib\n\n"
                "theorem mock_sub : True := by\n"
                "  trivial\n"
            )
        if t == "sorrifier_fill":
            code = prompt
            if "sorry" in code:
                return code.replace("sorry", "trivial", 1)
            return code
        if t == "sorrifier_fix":
            return (
                "import Mathlib\n\n"
                "theorem mock_fixed : True := by\n"
                "  trivial\n"
            )
        return f"[LOCAL_LLM_MOCK STUB] task={task!r}\n---\n{system}\n---\n{prompt[:500]}"

    def _post_json_urllib(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Local LLM HTTP {e.code}: {err_body}") from e
        return json.loads(raw)
