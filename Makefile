.PHONY: serve smoke loadtest sweep report site test

serve:
	docker compose up --build

smoke:
	docker compose -f docker-compose.yml -f docker-compose.smoke.yml up --build -d --wait router prometheus grafana
	docker compose -f docker-compose.yml -f docker-compose.smoke.yml --profile tools run --rm --build experiment-runner --url http://router:8080 --workload short --requests 8 --concurrency 2 --output /results/smoke/requests.jsonl

loadtest:
	uv run servebench-loadgen --url $${ROUTER_URL:-http://localhost:8080} --workload mixed --requests $${REQUESTS:-100} --concurrency $${CONCURRENCY:-8} --output results/latest/requests.jsonl

sweep:
	uv run servebench-experiment --config experiments/baseline.yaml --url $${ROUTER_URL:-http://localhost:8080} --prometheus-url $${PROMETHEUS_URL:-http://localhost:9090}

report:
	@if test -f results/report-input.json; then \
		uv run servebench-report results/report-input.json --report REPORT.md --figures figures; \
	else \
		echo "No results/report-input.json; retaining evidence-pending REPORT.md"; \
	fi

site:
	docker compose up --build web

test:
	uv run --extra dev pytest -q
