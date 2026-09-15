import httpx

from loadgen.tokenize import calibrate_requests
from servebench.workloads import WorkloadGenerator


async def test_calibration_uses_served_model_and_exact_prompt_lengths() -> None:
    models = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        import json

        body = json.loads(payload)
        models.append(body["model"])
        words = body["prompt"].split()
        tokens = [sum(word.encode()) % 50000 for word in words]
        return httpx.Response(200, json={"count": len(tokens), "tokens": tokens})

    specs = (
        WorkloadGenerator().generate("short", 1, 1)
        + WorkloadGenerator().generate("long-prefill", 1, 2)
        + WorkloadGenerator().generate("shared-prefix", 2, 3)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        calibrated = await calibrate_requests(specs, "http://vllm:8000", "custom/model", client)
    assert models and set(models) == {"custom/model"}
    assert [len(spec.prompt) for spec in calibrated] == [256, 4096, 4096, 4096]
    assert calibrated[2].prompt[:3968] == calibrated[3].prompt[:3968]
    assert calibrated[2].prompt[3968:] != calibrated[3].prompt[3968:]
