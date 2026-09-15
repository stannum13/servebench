"""Small OpenAI-compatible streaming backend used only for smoke tests."""

from __future__ import annotations

import asyncio
import json
import os
import zlib

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse


def create_app(token_delay: float = 0.001) -> FastAPI:
    app = FastAPI(title="servebench mock backend")

    @app.get("/healthz")
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            "vllm:kv_cache_usage_perc 0\n"
            "# TYPE vllm:gpu_cache_usage_perc gauge\n"
            "vllm:gpu_cache_usage_perc 0\n"
            "# TYPE vllm:num_preemptions_total counter\n"
            "vllm:num_preemptions_total 0\n"
        )

    @app.post("/tokenize")
    async def tokenize(request: Request) -> dict[str, object]:
        body = await request.json()
        words = str(body.get("prompt", "")).split()
        tokens = [zlib.crc32(word.encode()) % 50000 for word in words]
        return {"count": len(tokens), "tokens": tokens, "max_model_len": 32768}

    @app.post("/v1/completions")
    async def completions(request: Request) -> StreamingResponse:
        body = await request.json()
        count = int(body.get("max_tokens", 128))
        prompt = body.get("prompt", "")
        prompt_count = len(prompt) if isinstance(prompt, list) else len(str(prompt).split())

        async def events():
            for index in range(count):
                if token_delay:
                    await asyncio.sleep(token_delay)
                choice = {"text": f" t{index}", "token_ids": [index]}
                if index == 0:
                    choice["prompt_token_ids"] = list(range(prompt_count))
                yield f'data: {json.dumps({"choices": [choice]})}\n\n'
            usage = {
                "prompt_tokens": prompt_count,
                "completion_tokens": count,
            }
            yield f'data: {json.dumps({"choices": [], "usage": usage})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app(float(os.getenv("MOCK_TOKEN_DELAY", "0.001")))
