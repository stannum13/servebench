from router.policy import FifoPolicy, RequestFeatures, RouterSnapshot, SloPolicy, WorkerState


def snapshot(*workers: WorkerState) -> RouterSnapshot:
    return RouterSnapshot(workers=list(workers))


def test_fifo_always_admits_on_first_worker() -> None:
    decision = FifoPolicy().decide(RequestFeatures(256), snapshot(WorkerState("a", 99, 0.99)))
    assert decision.action == "admit"
    assert decision.worker == "a"


def test_slo_admits_immediately_below_thresholds() -> None:
    decision = SloPolicy().decide(RequestFeatures(256), snapshot(WorkerState("a", 2, 0.20)))
    assert decision.action == "admit"


def test_slo_delays_briefly_near_saturation() -> None:
    decision = SloPolicy().decide(RequestFeatures(256), snapshot(WorkerState("a", 20, 0.85)))
    assert decision.action == "delay"
    assert 0 < decision.delay_seconds <= 0.25


def test_slo_rejects_when_every_worker_is_hard_overloaded() -> None:
    decision = SloPolicy().decide(RequestFeatures(256), snapshot(WorkerState("a", 130, 0.97)))
    assert decision.action == "reject"
    assert "overload" in decision.reason


def test_slo_protects_kv_from_long_prompt() -> None:
    decision = SloPolicy().decide(RequestFeatures(4096), snapshot(WorkerState("a", 5, 0.82)))
    assert decision.action == "delay"
    assert "long prompt" in decision.reason


def test_slo_routes_to_healthy_alternate_worker() -> None:
    state = snapshot(WorkerState("a", 80, 0.94), WorkerState("b", 3, 0.30))
    decision = SloPolicy().decide(RequestFeatures(256), state)
    assert decision.action == "admit"
    assert decision.worker == "b"
    assert "alternate" in decision.reason
