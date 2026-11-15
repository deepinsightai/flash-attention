"""Small Lambda Cloud API client for launching single-GPU benchmark instances.

API key: set the LAMBDA_API_KEY environment variable, or put the key alone in
~/.lambda_cloud/api_key. Never commit it. Create keys at https://cloud.lambda.com/api-keys.

    python -m scripts.lambda_cloud types --gpu a100            # prices + regions with capacity now
    python -m scripts.lambda_cloud setup-ssh                   # create local key, upload public half
    python -m scripts.lambda_cloud launch --type gpu_1x_a100_sxm4 --region us-east-1 --yes
    python -m scripts.lambda_cloud list                        # running instances, IPs, cost so far
    python -m scripts.lambda_cloud terminate <id> --yes        # stop billing

Launch and terminate cost money or destroy state, so both refuse to run without --yes.
API reference: https://docs.lambda.com/api/cloud
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

BASE = "https://cloud.lambda.com/api/v1"
KEY_FILE = Path.home() / ".lambda_cloud" / "api_key"
SSH_KEY = Path.home() / ".ssh" / "lambda_ed25519"
SSH_KEY_NAME = "comp9060-benchmark"
TIMEOUT = 30


class LambdaError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise LambdaError(
            "No API key. Set LAMBDA_API_KEY or write the key to ~/.lambda_cloud/api_key "
            "(create one at https://cloud.lambda.com/api-keys)."
        )
    return key


class Client:
    def __init__(self, key: str | None = None, session: requests.Session | None = None):
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key or api_key()}", "accept": "application/json"})

    def _req(self, method: str, path: str, body: dict | None = None):
        r = self.s.request(method, BASE + path, json=body, timeout=TIMEOUT)
        try:
            payload = r.json()
        except ValueError:
            payload = {}
        if r.status_code >= 400:
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            raise LambdaError(f"{method} {path} -> {r.status_code}: {err.get('code', '')} {err.get('message', r.text[:200])}")
        return payload.get("data", payload)

    def instance_types(self) -> dict:
        return self._req("GET", "/instance-types")

    def instances(self) -> list:
        return self._req("GET", "/instances")

    def ssh_keys(self) -> list:
        return self._req("GET", "/ssh-keys")

    def add_ssh_key(self, name: str, public_key: str) -> dict:
        return self._req("POST", "/ssh-keys", {"name": name, "public_key": public_key})

    def launch(self, type_name: str, region: str, ssh_key: str, name: str) -> list:
        body = {"region_name": region, "instance_type_name": type_name, "ssh_key_names": [ssh_key], "name": name}
        return self._req("POST", "/instance-operations/launch", body)["instance_ids"]

    def terminate(self, ids: list[str]) -> list:
        return self._req("POST", "/instance-operations/terminate", {"instance_ids": ids})["terminated_instances"]


def single_gpu_types(types: dict, gpu: str | None = None) -> list[dict]:
    """Flatten /instance-types into rows for 1-GPU types, cheapest first."""
    rows = []
    for name, entry in types.items():
        it = entry["instance_type"]
        if it["specs"]["gpus"] != 1:
            continue
        if gpu and gpu.lower() not in (name + it["gpu_description"]).lower():
            continue
        rows.append({
            "name": name,
            "gpu": it["gpu_description"],
            "usd_per_hour": it["price_cents_per_hour"] / 100,
            "regions": [r["name"] for r in entry.get("regions_with_capacity_available", [])],
        })
    return sorted(rows, key=lambda r: r["usd_per_hour"])


def cmd_types(c: Client, a):
    rows = single_gpu_types(c.instance_types(), a.gpu)
    if not rows:
        print("no matching single-GPU instance types")
    for r in rows:
        avail = ", ".join(r["regions"]) if r["regions"] else "NO CAPACITY NOW"
        hours = 50 / r["usd_per_hour"] if r["usd_per_hour"] else float("inf")
        print(f"{r['name']:24s} {r['gpu']:28s} ${r['usd_per_hour']:5.2f}/h  ({hours:4.1f} h per $50)  {avail}")


def cmd_setup_ssh(c: Client, a):
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", str(SSH_KEY), "-N", "", "-C", SSH_KEY_NAME], check=True)
        print(f"created {SSH_KEY} (no passphrase; keep this file private)")
    pub = SSH_KEY.with_suffix(".pub").read_text(encoding="utf-8").strip()
    existing = {k["name"]: k for k in c.ssh_keys()}
    if SSH_KEY_NAME in existing:
        same = existing[SSH_KEY_NAME]["public_key"].split()[:2] == pub.split()[:2]
        print(f"key '{SSH_KEY_NAME}' already on Lambda" + ("" if same else " BUT DIFFERS from the local key"))
        return
    c.add_ssh_key(SSH_KEY_NAME, pub)
    print(f"uploaded public key as '{SSH_KEY_NAME}'")


def cmd_list(c: Client, a):
    inst = c.instances()
    if not inst:
        print("no running instances (nothing is billing)")
    now = dt.datetime.now(dt.timezone.utc)
    for i in inst:
        it = i["instance_type"]
        rate = it["price_cents_per_hour"] / 100
        line = f"{i['id']}  {i.get('status', '?'):9s} {it['name']:22s} {i['region']['name']:12s} ip={i.get('ip') or '-'}  ${rate:.2f}/h"
        if i.get("created"):
            started = dt.datetime.fromisoformat(i["created"].replace("Z", "+00:00"))
            hours = (now - started).total_seconds() / 3600
            line += f"  up {hours:.2f} h, ~${hours * rate:.2f}"
        print(line)
        if i.get("ip"):
            print(f"    ssh -i {SSH_KEY} ubuntu@{i['ip']}")


def cmd_launch(c: Client, a):
    rows = {r["name"]: r for r in single_gpu_types(c.instance_types())}
    if a.type not in rows:
        raise LambdaError(f"{a.type!r} is not a single-GPU type; run 'types' to see options")
    r = rows[a.type]
    if a.region not in r["regions"]:
        raise LambdaError(f"no capacity for {a.type} in {a.region} right now; available: {r['regions'] or 'none'}")
    print(f"launch {a.type} ({r['gpu']}) in {a.region} at ${r['usd_per_hour']:.2f}/h, billed until terminated")
    if not a.yes:
        print("dry run: add --yes to launch")
        return
    ids = c.launch(a.type, a.region, a.ssh_key, a.name)
    print("launched:", ", ".join(ids))
    print("run 'list' in a minute for the IP; run 'terminate <id> --yes' when done")


def cmd_terminate(c: Client, a):
    if not a.yes:
        print(f"would terminate {a.ids}; this deletes the instance and its local disk. Add --yes.")
        return
    for t in c.terminate(a.ids):
        print(f"{t['id']} -> {t.get('status')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("types"); t.add_argument("--gpu", help="filter, e.g. a100, h100, b200")
    sub.add_parser("setup-ssh")
    sub.add_parser("list")
    l = sub.add_parser("launch")
    l.add_argument("--type", required=True); l.add_argument("--region", required=True)
    l.add_argument("--ssh-key", default=SSH_KEY_NAME); l.add_argument("--name", default="comp9060-bench")
    l.add_argument("--yes", action="store_true")
    k = sub.add_parser("terminate"); k.add_argument("ids", nargs="+"); k.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)
    try:
        c = Client()
        {"types": cmd_types, "setup-ssh": cmd_setup_ssh, "list": cmd_list, "launch": cmd_launch,
         "terminate": cmd_terminate}[a.cmd](c, a)
    except LambdaError as e:
        print("error:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
