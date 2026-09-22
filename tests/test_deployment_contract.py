import json
from pathlib import Path

import yaml


def test_compose_contains_required_services_and_gpu_contract() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = compose["services"]
    required = {"router", "vllm", "prometheus", "grafana", "dcgm-exporter", "experiment-runner"}
    assert required <= set(services)
    devices = services["vllm"]["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["capabilities"] == ["gpu"]
    assert "healthcheck" in services["router"] and "healthcheck" in services["vllm"]
    assert any("results" in volume for volume in services["experiment-runner"]["volumes"])
    for service in ("router", "vllm", "prometheus", "grafana", "dcgm-exporter"):
        assert all(str(port).startswith("127.0.0.1:") for port in services[service]["ports"])
    assert "${GRAFANA_HOST_PORT:-3000}" in services["grafana"]["ports"][0]


def test_makefile_exposes_operator_commands() -> None:
    text = Path("Makefile").read_text()
    targets = ("serve", "smoke", "loadtest", "sweep", "engine-sweep", "compare", "report", "site")
    for target in targets:
        assert f"{target}:" in text
    smoke_recipe = text.split("smoke:", 1)[1].split("\n\n", 1)[0]
    assert "run --rm --build experiment-runner" in smoke_recipe
    report_recipe = text.split("report:", 1)[1].split("\n\n", 1)[0]
    assert "test -f" in report_recipe


def test_prometheus_scrapes_router_vllm_and_dcgm() -> None:
    config = yaml.safe_load(Path("configs/prometheus.yml").read_text())
    jobs = {item["job_name"] for item in config["scrape_configs"]}
    assert {"router", "vllm", "dcgm"} <= jobs


def test_docker_build_copies_package_readme_before_install() -> None:
    lines = Path("docker/Dockerfile").read_text().splitlines()
    readme_copy = next(index for index, line in enumerate(lines) if "README.md" in line)
    install = next(index for index, line in enumerate(lines) if "pip install" in line)
    assert readme_copy < install


def test_committed_results_cover_decision_and_timing_contract() -> None:
    comparison = json.loads(Path("results/mock-scheduler/comparison.json").read_text())
    assert {"completion_ratio", "policy_totals", "provenance"} <= comparison.keys()
    assert {"requests", "successful", "rejections", "timeouts", "failures"} <= (
        comparison["policy_totals"]["slo"].keys()
    )

    summary = json.loads(Path("results/smoke/summary.json").read_text())
    assert {"inter_token_latency_ms", "tpot_ms"} <= summary.keys()

    runs = json.loads(Path("results/mock-scheduler/runs.json").read_text())
    required = {
        "model", "workload", "requested_requests", "seed",
        "cache_isolation", "cache_state_initial", "execution_order",
    }
    assert runs and all(required <= row.keys() for row in runs)


def test_three_required_graph_artifacts_exist() -> None:
    assert all(Path("figures", name).is_file() for name in (
        "saturation.png", "latency-decomposition.png", "scheduler-comparison.png",
    ))
