from pathlib import Path


def test_readme_documents_project_contract() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    for phrase in ("make serve", "make smoke", "make loadtest", "make sweep", "make report"):
        assert phrase in text
    assert "MODEL" in text
    assert "```mermaid" in text
    assert "results/" in text
    assert text.count("figures/") >= 3
    assert "without a GPU" in text


def test_state_and_report_do_not_claim_unmeasured_gpu_improvement() -> None:
    state = Path("BENCH_STATE.md").read_text(encoding="utf-8").lower()
    report = Path("REPORT.md").read_text(encoding="utf-8").lower()
    assert "pending" in state
    assert "no optimization claim" in report
