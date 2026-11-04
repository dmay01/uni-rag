import json

import httpx
import pytest

from tests.test_m1 import transport
from uni_rag.energy import PowerSampler, Sample, integrate, macmon_sample, powermetrics_samples
from uni_rag.generate import ask
from uni_rag.ingest.pipeline import ingest
from uni_rag.ollama import Ollama
from uni_rag.stats import stats


def test_integration_requires_full_coverage_and_no_gaps():
    samples = [Sample(0, 2), Sample(1, 4), Sample(2, 6)]
    assert integrate(samples, 0.5, 1.5, 1.1) == (4.0, 1000.0)
    assert integrate(samples, -0.1, 1, 1.1) == (None, None)
    assert integrate(samples, 1, 2.1, 1.1) == (None, None)
    assert integrate([Sample(0, 2), Sample(2, 6)], 0.5, 1.5, 1.1) == (None, None)


def test_macmon_schema_requires_all_components():
    result = macmon_sample(
        {"timestamp": "2026-10-30T12:00:00+00:00", "cpu_power": 2, "gpu_power": 3, "ane_power": 0.5}
    )
    assert result.watts == 5.5
    with pytest.raises(KeyError):
        macmon_sample({"timestamp": "2026-10-30T12:00:00+00:00", "cpu_power": 2})
    with pytest.raises(ValueError):
        macmon_sample(
            {
                "timestamp": "2026-10-30T12:00:00+00:00",
                "cpu_power": -2,
                "gpu_power": 3,
                "ane_power": 0,
            }
        )


def test_powermetrics_units_and_incomplete_samples(cfg):
    cfg.data_dir.mkdir()
    cfg.power_log.write_text(
        "*** Sampled system activity (Fri Oct 30 12:00:00 2026 +0100) (200ms elapsed) ***\n"
        "CPU Power: 2000 mW\nGPU Power: 500 mW\nANE Power: 100 mW\n"
        "*** Sampled system activity (Fri Oct 30 12:00:01 2026 +0100) (200ms elapsed) ***\n"
        "CPU Power: 2 W\nGPU Power: 1 W\n"
    )
    samples = powermetrics_samples(cfg.power_log)
    assert len(samples) == 1
    assert samples[0].watts == pytest.approx(2.6)


def test_auto_falls_back_only_for_log_with_query_coverage(cfg, monkeypatch):
    import uni_rag.energy as energy

    monkeypatch.setattr(PowerSampler, "_start_macmon", lambda self: None)
    cfg.data_dir.mkdir()
    cfg.power_log.touch()
    monkeypatch.setattr(energy, "powermetrics_samples", lambda path: [Sample(10, 4), Sample(11, 4)])
    with PowerSampler(cfg, "auto") as sampler:
        result = sampler.measure(10.1, 10.9)
        assert result["energy_backend"] == "powermetrics"
        assert result["energy_max_allowed_gap_ms"] == 1250
        assert result["energy_gross_joules"] == pytest.approx(3.2)
        assert result["energy_above_baseline_joules"] is None
        sampler.baselines["powermetrics"] = 2
        assert sampler.measure(10.1, 10.9)["energy_above_baseline_joules"] == pytest.approx(1.6)
        ticks = iter([0, 20])
        monkeypatch.setattr(energy.time, "monotonic", lambda: next(ticks))
        result = sampler.measure(100, 101)
        assert result["energy_backend"] == "none" and result["energy_gross_joules"] is None


def test_complete_window_survives_later_sampler_failure(cfg, monkeypatch):

    monkeypatch.setattr(PowerSampler, "_start_macmon", lambda self: None)
    with PowerSampler(cfg, "macmon") as sampler:
        sampler.samples.extend([Sample(0, 2), Sample(0.2, 2)])
        sampler.failed = True
        result = sampler.measure(0.01, 0.19)
        incomplete = sampler.measure(0.01, 0.3)
    assert result["energy_gross_joules"] == pytest.approx(0.36)
    assert incomplete["energy_gross_joules"] is None


def test_query_metrics_and_power_failure_are_nonfatal(cfg, monkeypatch):
    def token_transport(request):
        response = transport(request)
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                text=response.text.replace(
                    '"eval_count":8', '"eval_count":8,"eval_duration":500000000'
                ),
            )
        return response

    with Ollama(cfg, httpx.MockTransport(token_transport)) as client:
        ingest(cfg, ollama=client)
        with PowerSampler(cfg, "none") as sampler:
            monkeypatch.setattr(
                sampler,
                "measure",
                lambda *args: (_ for _ in ()).throw(RuntimeError("sensor stopped")),
            )
            answer = ask("What is hybrid retrieval?", cfg, client, sampler=sampler)
    metric = answer.metrics
    assert metric["think"] is False and metric["energy_backend"] == "none"
    assert metric["tokens_per_second"] == 16
    assert metric["total_seconds"] >= metric["ttft_seconds"] >= metric["retrieval_seconds"]
    recorded = json.loads(next((cfg.data_dir / "metrics").glob("*.jsonl")).read_text())
    assert recorded == metric
    assert "What is hybrid" not in json.dumps(recorded)
    assert recorded["energy_gross_joules"] is None
    assert stats(cfg)["queries_by_status"] == {"ok": 1}


def test_failed_query_is_measured(cfg):
    def failed(request):
        if request.url.path == "/api/chat":
            return httpx.Response(200, text='{"message":{"content":"partial"}}\n')
        return transport(request)

    with Ollama(cfg, httpx.MockTransport(failed)) as client:
        ingest(cfg, ollama=client)
        with pytest.raises(RuntimeError, match="final event"):
            ask("What is hybrid retrieval?", cfg, client)
    metric = json.loads(next((cfg.data_dir / "metrics").glob("*.jsonl")).read_text())
    assert metric["status"] == "error"
    assert metric["energy_backend"] == "none"


def test_ask_never_measures_idle_baseline(cfg, monkeypatch):
    monkeypatch.setattr(
        PowerSampler,
        "measure_idle",
        lambda *args: pytest.fail("ask must not measure idle baseline"),
    )
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
        ask("What is hybrid retrieval?", cfg, client)


def test_think_forwarding_and_visible_ttft(cfg):
    def thinking(request):
        if request.url.path != "/api/chat":
            return transport(request)
        assert json.loads(request.content)["think"] is True
        return httpx.Response(
            200,
            text='{"message":{"thinking":"internal"}}\n'
            '{"message":{"content":"Dense vectors and BM25 [1]."}}\n'
            '{"done":true,"prompt_eval_count":80,"eval_count":12}\n',
        )

    with Ollama(cfg, httpx.MockTransport(thinking)) as client:
        ingest(cfg, ollama=client)
        answer = ask("What is hybrid retrieval?", cfg, client, think=True)
    assert answer.metrics["think"] is True
    assert answer.metrics["first_model_token_seconds"] <= answer.metrics["ttft_seconds"]
    assert "internal" not in answer.text
