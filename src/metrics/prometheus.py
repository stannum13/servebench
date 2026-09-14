"""Low-cardinality Prometheus instruments for admission decisions."""

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest


class RouterMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.queue = Gauge(
            "servebench_router_queue_depth", "Requests assigned to a worker", ["worker"],
            registry=self.registry,
        )
        self.kv = Gauge(
            "servebench_worker_kv_cache_usage_ratio", "Worker KV cache utilization", ["worker"],
            registry=self.registry,
        )
        self.decisions = Counter(
            "servebench_admission_decisions_total", "Admission decisions", ["action"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)

