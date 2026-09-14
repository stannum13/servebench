"""Small OpenAI-compatible streaming backend used only for smoke tests."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def create_app(token_delay: float = 0.001) -> FastAPI:
    app = FastAPI(title="servebench mock backend")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request) -> StreamingResponse:
        body = await request.json()
        count = int(body.get("max_tokens", 128))

        async def events():
            for index in range(count):
                if token_delay:
                    await asyncio.sleep(token_delay)
                yield f'data: {json.dumps({"choices": [{"text": f" t{index}"}]})}\n\n'
            usage = {
                "prompt_tokens": len(str(body.get("prompt", "")).split()),
                "completion_tokens": count,
            }
            yield f'data: {json.dumps({"choices": [], "usage": usage})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app(float(os.getenv("MOCK_TOKEN_DELAY", "0.001")))
