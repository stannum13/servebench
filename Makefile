.PHONY: serve smoke loadtest sweep report test

serve:
	docker compose up --build

smoke:
	docker compose -f docker-compose.yml -f docker-compose.smoke.yml up --build -d --wait router prometheus grafana
	docker compose -f docker-compose.yml -f docker-compose.smoke.yml --profile tools run --rm experiment-runner --url http://router:8080 --workload short --requests 8 --concurrency 2 --output /results/smoke/requests.jsonl

loadtest:
	uv run servebench-loadgen --url $${ROUTER_URL:-http://localhost:8080} --workload mixed --requests $${REQUESTS:-100} --concurrency $${CONCURRENCY:-8} --output results/latest/requests.jsonl

sweep:
	uv run servebench-experiment results/*/summary.json --output results/experiment-manifest.json

report:
	uv run servebench-report results/report-input.json --report REPORT.md --figures figures

test:
	uv run --extra dev pytest -q
