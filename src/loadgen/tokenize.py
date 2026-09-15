"""Preflight exact-length prompts using the active vLLM tokenizer API."""

from __future__ import annotations

from dataclasses import replace

import httpx

from servebench.workloads import RequestSpec


async def calibrate_requests(
    specs: list[RequestSpec],
    tokenizer_url: str,
    model: str,
    client: httpx.AsyncClient | None = None,
) -> list[RequestSpec]:
    own_client = client is None
    active_client = client or httpx.AsyncClient(timeout=60)
    prefix_cache: dict[str, list[int]] = {}

    async def enough_tokens(text: str, target: int) -> list[int]:
        candidate = text
        for _ in range(4):
            response = await active_client.post(
                tokenizer_url.rstrip("/") + "/tokenize",
                json={"model": model, "prompt": candidate},
            )
            response.raise_for_status()
            tokens = response.json().get("tokens")
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                raise ValueError("tokenizer response is missing integer token IDs")
            if len(tokens) >= target:
                return tokens[:target]
            candidate = f"{candidate} {text}"
        raise ValueError(f"tokenizer produced fewer than {target} tokens for workload")

    try:
        calibrated: list[RequestSpec] = []
        for spec in specs:
            if not isinstance(spec.prompt, str):
                calibrated.append(spec)
                continue
            if spec.kind == "shared-prefix":
                prefix, unique = spec.prompt.split(" UNIQUE_SUFFIX ", 1)
                if prefix not in prefix_cache:
                    prefix_cache[prefix] = await enough_tokens(prefix, 3968)
                suffix = await enough_tokens("UNIQUE_SUFFIX " + unique, 128)
                tokens = prefix_cache[prefix] + suffix
            else:
                tokens = await enough_tokens(spec.prompt, spec.prompt_tokens)
            calibrated.append(replace(spec, prompt=tokens, prompt_tokens=len(tokens)))
        return calibrated
    finally:
        if own_client:
            await active_client.aclose()
