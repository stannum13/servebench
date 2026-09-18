"""Low-cardinality Prometheus instruments for admission decisions."""

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest


class RouterMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.queue = Gauge(
            "servebench_router_queue_depth", "Requests assigned to a worker (in-flight)",
            ["worker"],
            registry=self.registry,
        )
        self.backend_waiting = Gauge(
            "servebench_backend_waiting_requests", "vLLM waiting requests sampled per worker",
            ["worker"], registry=self.registry,
        )
        self.pressure_fresh = Gauge(
            "servebench_worker_pressure_telemetry_fresh",
            "Whether KV and backend waiting telemetry is within its freshness TTL",
            ["worker"], registry=self.registry,
        )
        self.kv = Gauge(
            "servebench_worker_kv_cache_usage_ratio", "Worker KV cache utilization", ["worker"],
            registry=self.registry,
        )
        self.prefix_hits = Gauge(
            "servebench_worker_prefix_cache_hit_ratio",
            "Worker prefix cache hit ratio",
            ["worker"],
            registry=self.registry,
        )
        self.preemptions = Gauge(
            "servebench_worker_preemptions_total",
            "Worker-reported cumulative preemptions",
            ["worker"],
            registry=self.registry,
        )
        self.decisions = Counter(
            "servebench_admission_decisions_total", "Admission decisions", ["action"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
