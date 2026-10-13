"""LLM client protocol — librarium's only integration point.

librarium needs exactly one capability from a language model: send a
prompt, get text back.  Implement the ``LLMClient`` protocol (a single
async ``complete`` method) to plug in any backend, or use the bundled
``OpenAICompatClient`` which speaks the OpenAI chat-completions API
(vLLM, Ollama, OpenAI, RunPod, together.ai, ...).
"""

from __future__ import annotations

import numbers
import re
from typing import Protocol, runtime_checkable

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_thinking(content: str) -> tuple[str, str | None]:
    """Split <think>...</think> reasoning out of a model response.

    Returns (cleaned_content, thinking_text); thinking_text is None
    when no think tags are present.
    """
    match = _THINK_RE.search(content)
    if not match:
        return content.strip(), None
    thinking = match.group(1).strip()
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned, thinking


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can complete a prompt.

    Implementations must strip provider-specific artifacts (e.g.
    <think> blocks) before returning — callers receive clean text.
    """

    async def complete(
        self, system: str, user: str, *, think: bool = False,
        temperature: float = 0.1,
    ) -> str: ...


class OpenAICompatClient:
    """LLMClient for any OpenAI-compatible chat-completions endpoint.

    Works with vLLM, Ollama (/v1), OpenAI, and most hosted providers.

    Args:
        base_url: e.g. "http://localhost:8000/v1"
        model: served model id (see GET {base_url}/models)
        api_key: bearer token; many local servers accept anything
        think: passed per-call; when the server supports Qwen-style
            ``chat_template_kwargs.enable_thinking`` it is forwarded,
            and any <think> blocks are stripped from the reply either way
        timeout: per-request timeout in seconds
        max_connections: size of the HTTP connection pool. This is the
            other half of ``SourceLibrarian(max_concurrent_prefixes=...)``:
            capping how many calls a librarian starts does nothing if the
            pool is narrower, because the requests just queue there
            instead. Keep this at or above the librarian's setting. Every
            connection may be kept alive, since librarium talks to one
            endpoint in bursts and reconnecting per call is pure overhead.
            ``0`` means unbounded, matching ``max_concurrent_prefixes`` —
            so the librarian's unbounded setting has a pool setting to
            pair with. httpx spells unbounded ``None``, and takes a literal
            0 as a pool that admits nothing; with no default pool_timeout
            that is not an error but a permanent hang, so 0 is translated
            here rather than passed through. ``None`` is accepted as a
            synonym for 0, since that is httpx's spelling and predates
            this argument's.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_tokens: int | None = None,
        max_connections: int | None = 32,
    ):
        import httpx  # deferred so the core package works without it

        # None is httpx's own spelling of unbounded and was accepted before
        # the 0 spelling existed; keep it working rather than letting the
        # comparison below raise TypeError at people who already pass it.
        if max_connections is None:
            max_connections = 0
        # numbers.Integral rather than int, so a pool size that came out of
        # a numpy computation is accepted; numpy is a hard dependency of
        # this package. bool is still refused — it registers as Integral,
        # but True as a pool size is a mistake rather than a one-connection
        # pool. (operator.index() would accept it, hence not using it.)
        if (not isinstance(max_connections, numbers.Integral)
                or isinstance(max_connections, bool)):
            raise TypeError(
                f"max_connections must be an int, got "
                f"{type(max_connections).__name__} — pass 0 for an "
                f"unbounded pool"
            )
        if max_connections < 0:
            raise ValueError(
                f"max_connections must be >= 0, got {max_connections} — "
                f"pass 0 for an unbounded pool"
            )
        # Normalise at the boundary: this value is stored, handed to
        # httpx.Limits, and compared against the librarian's cap by
        # _warn_if_pool_is_narrower, none of which should have to know a
        # numpy scalar from an int.
        max_connections = int(max_connections)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.max_connections = max_connections
        pool = max_connections or None  # 0 → unbounded, as httpx spells it
        self.limits = httpx.Limits(
            max_connections=pool,
            max_keepalive_connections=pool,
        )
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            limits=self.limits,
        )

    async def complete(
        self, system: str, user: str, *, think: bool = False,
        temperature: float = 0.1,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        resp = await self._client.post(
            f"{self.base_url}/chat/completions", json=payload
        )
        if resp.status_code == 400 and "chat_template_kwargs" in resp.text:
            # Server doesn't support the kwarg (e.g. OpenAI) — retry without
            payload.pop("chat_template_kwargs")
            resp = await self._client.post(
                f"{self.base_url}/chat/completions", json=payload
            )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        cleaned, _ = parse_thinking(content)
        return cleaned

    async def aclose(self) -> None:
        await self._client.aclose()
