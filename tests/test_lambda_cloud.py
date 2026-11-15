"""Offline tests for scripts/lambda_cloud.py: no network, no API key, nothing launched."""

import pytest

pytest.importorskip("requests")  # the Lambda client runs on the local PC; GPU machines need not have it

from scripts import lambda_cloud as lc  # noqa: E402

TYPES = {
    "gpu_1x_a100_sxm4": {
        "instance_type": {"name": "gpu_1x_a100_sxm4", "description": "1x A100 (40 GB SXM4)",
                          "gpu_description": "A100 (40 GB SXM4)", "price_cents_per_hour": 199,
                          "specs": {"vcpus": 30, "memory_gib": 200, "storage_gib": 512, "gpus": 1}},
        "regions_with_capacity_available": [{"name": "us-east-1", "description": "Virginia"}],
    },
    "gpu_1x_h100_pcie": {
        "instance_type": {"name": "gpu_1x_h100_pcie", "description": "1x H100 (80 GB PCIe)",
                          "gpu_description": "H100 (80 GB PCIe)", "price_cents_per_hour": 249,
                          "specs": {"vcpus": 26, "memory_gib": 200, "storage_gib": 1024, "gpus": 1}},
        "regions_with_capacity_available": [],
    },
    "gpu_8x_a100": {
        "instance_type": {"name": "gpu_8x_a100", "description": "8x A100", "gpu_description": "A100 (40 GB SXM4)",
                          "price_cents_per_hour": 1592, "specs": {"vcpus": 124, "memory_gib": 1800,
                                                                  "storage_gib": 6144, "gpus": 8}},
        "regions_with_capacity_available": [{"name": "us-west-1", "description": "California"}],
    },
}


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.headers, self.routes, self.calls = {}, routes, []

    def request(self, method, url, json=None, timeout=None):
        path = url.removeprefix(lc.BASE)
        self.calls.append((method, path, json))
        status, payload = self.routes[(method, path)]
        return FakeResponse(status, payload)


def client(routes):
    s = FakeSession(routes)
    return lc.Client(key="test-key", session=s), s


def test_single_gpu_filter_sorted_and_excludes_multi_gpu():
    rows = lc.single_gpu_types(TYPES)
    assert [r["name"] for r in rows] == ["gpu_1x_a100_sxm4", "gpu_1x_h100_pcie"]
    assert lc.single_gpu_types(TYPES, "h100")[0]["regions"] == []


def test_auth_header_and_error_mapping():
    c, s = client({("GET", "/instances"): (401, {"error": {"code": "global/invalid-api-key", "message": "bad"}})})
    assert s.headers["Authorization"] == "Bearer test-key"
    with pytest.raises(lc.LambdaError, match="invalid-api-key"):
        c.instances()


def test_launch_is_dry_run_without_yes(capsys):
    c, s = client({("GET", "/instance-types"): (200, {"data": TYPES})})
    lc.cmd_launch(c, type("A", (), {"type": "gpu_1x_a100_sxm4", "region": "us-east-1", "ssh_key": "k",
                                     "name": "n", "yes": False})())
    assert "dry run" in capsys.readouterr().out
    assert all(call[1] != "/instance-operations/launch" for call in s.calls)


def test_launch_refuses_region_without_capacity():
    c, _ = client({("GET", "/instance-types"): (200, {"data": TYPES})})
    with pytest.raises(lc.LambdaError, match="no capacity"):
        lc.cmd_launch(c, type("A", (), {"type": "gpu_1x_h100_pcie", "region": "us-east-1", "ssh_key": "k",
                                         "name": "n", "yes": True})())


def test_launch_with_yes_posts_request():
    c, s = client({("GET", "/instance-types"): (200, {"data": TYPES}),
                   ("POST", "/instance-operations/launch"): (200, {"data": {"instance_ids": ["abc"]}})})
    lc.cmd_launch(c, type("A", (), {"type": "gpu_1x_a100_sxm4", "region": "us-east-1", "ssh_key": "k",
                                     "name": "n", "yes": True})())
    assert s.calls[-1] == ("POST", "/instance-operations/launch",
                           {"region_name": "us-east-1", "instance_type_name": "gpu_1x_a100_sxm4",
                            "ssh_key_names": ["k"], "name": "n"})


def test_terminate_requires_yes(capsys):
    c, s = client({})
    lc.cmd_terminate(c, type("A", (), {"ids": ["abc"], "yes": False})())
    assert "Add --yes" in capsys.readouterr().out and not s.calls


def test_missing_key_message(monkeypatch, tmp_path):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.setattr(lc, "KEY_FILE", tmp_path / "none")
    with pytest.raises(lc.LambdaError, match="No API key"):
        lc.api_key()
