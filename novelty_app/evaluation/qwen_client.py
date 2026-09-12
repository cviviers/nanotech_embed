from __future__ import annotations

import copy
import json
import sys
from typing import Any, Dict, List, Optional

import requests

try:
    from novelty_app.agents.observability import observe_current
except Exception:  # pragma: no cover
    from agents.observability import observe_current  # type: ignore


def _print_local_qwen_error(message: str) -> None:
    print(f"[QwenClient] {message}", file=sys.stderr, flush=True)


class QwenClient:
    """HTTP client for the local Qwen embedding + reranker service."""

    def __init__(self, base_url: str = "http://0.0.0.0:8000", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._response_cache: Dict[str, Dict[str, Any]] = {}

    def _cache_key(self, path: str, payload: Dict[str, Any]) -> str:
        return path + ":" + json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = self._cache_key(path, payload)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        input_summary: Dict[str, Any] = {
            "path": path,
            "base_url": self.base_url,
        }
        if path == "/embed":
            input_summary.update(
                {
                    "n_texts": len(payload.get("texts") or []),
                    "instruction": payload.get("instruction"),
                    "normalize": payload.get("normalize"),
                }
            )
        else:
            input_summary.update(
                {
                    "query": str(payload.get("query") or "")[:500],
                    "n_documents": len(payload.get("documents") or []),
                    "top_k": payload.get("top_k"),
                }
            )

        observation_type = "embedding" if path == "/embed" else "retriever"
        observation_name = "qwen_embed" if path == "/embed" else "qwen_rank"

        with observe_current(
            name=observation_name,
            as_type=observation_type,
            input_payload=input_summary,
            metadata={"path": path},
            model="qwen-local",
        ) as observation:
            try:
                resp = requests.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                _print_local_qwen_error(f"POST {path} request error: {exc}")
                raise
            if not resp.ok:
                detail = None
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text.strip() or None
                message = f"POST {path} failed with HTTP {resp.status_code}: {detail}"
                _print_local_qwen_error(message)
                raise RuntimeError(message)
            try:
                data = resp.json()
            except ValueError as exc:
                body = resp.text.strip()
                body_preview = body[:500] + ("..." if len(body) > 500 else "")
                message = f"POST {path} returned invalid JSON: {body_preview or '<empty response>'}"
                _print_local_qwen_error(message)
                raise RuntimeError(message) from exc

            output_summary = {"path": path}
            if path == "/embed":
                output_summary["n_embeddings"] = len(data.get("embeddings") or [])
            else:
                output_summary["n_results"] = len(data.get("results") or [])
            observation.update(output=output_summary)
            self._response_cache[cache_key] = copy.deepcopy(data)
            return data

    def embed(
        self,
        texts: List[str],
        *,
        instruction: Optional[str] = None,
        normalize: bool = True,
    ) -> List[List[float]]:
        data = self._post(
            "/embed",
            {"texts": texts, "instruction": instruction, "normalize": normalize},
        )
        return data.get("embeddings", [])

    def rank(
        self,
        *,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        return_embedding_similarity: bool = True,
        normalize_embeddings: bool = True,
        require_reranker: bool = False,
    ) -> List[Dict[str, Any]]:
        data = self._post(
            "/rank",
            {
                "query": query,
                "documents": documents,
                "instruction": instruction,
                "top_k": top_k,
                "return_embedding_similarity": return_embedding_similarity,
                "normalize_embeddings": normalize_embeddings,
            },
        )
        if require_reranker and data.get("used_embedding_fallback") is not False:
            raise RuntimeError("Strict matching requires reranker execution and fallback metadata; restart the updated Qwen service")
        return data.get("results", [])
