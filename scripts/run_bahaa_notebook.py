"""Run the code cells of Bahaa's Colab notebook locally, headless, for a compatibility check.

Reads the notebook from git (origin/bahaa-blackwell-benchmark), drops shell magics such as
`!nvidia-smi`, uses a non-interactive matplotlib backend, and executes the cells in order.
If PyTorch's FLASH_ATTENTION backend is missing (Windows builds), pass --fallback to substitute
EFFICIENT_ATTENTION so the rest of the notebook can still run. Results then are NOT Flash SDPA.
"""

import argparse
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/bahaa-blackwell-benchmark")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--max-n", type=int, default=0, help="skip cells whose output we cannot afford (unused)")
    args = ap.parse_args()

    raw = subprocess.check_output(["git", "show", f"{args.ref}:blackwell_sdpa_smoke_test.ipynb"])
    nb = json.loads(raw.decode("utf-8"))
    ns: dict = {"__name__": "__main__"}
    if args.fallback:
        from torch.nn.attention import SDPBackend

        SDPBackend.FLASH_ATTENTION = SDPBackend.EFFICIENT_ATTENTION  # type: ignore[misc]
    failed = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        if not src.strip():
            continue
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            print(f"cell {i:2d} ok")
        except BaseException as e:  # noqa: BLE001
            failed += 1
            print(f"cell {i:2d} FAILED: {type(e).__name__}: {str(e)[:160]}")
    print(f"done, {failed} failed cell(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
