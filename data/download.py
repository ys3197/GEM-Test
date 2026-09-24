"""
Download raw Amazon Reviews 2023 files for the configured domains.

Uses hf_hub_download rather than datasets.load_dataset on purpose: the dataset
repo ships a loading script, which recent `datasets` versions refuse to execute
without trust_remote_code. Pulling the jsonl files directly keeps the pipeline
explicit, cached, and reproducible — and makes it obvious exactly which bytes
the experiments are built on.

Usage:
    python -m data.download                 # all configured domains
    python -m data.download --smoke         # two small domains, for pipeline work
    python -m data.download --domains Digital_Music
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from config import DOMAINS, DOMAINS_SMOKE, HF_REPO, RAW_DIR  # noqa: E402


def download_domain(domain: str, with_meta: bool = True) -> dict[str, str]:
    """Fetch the review file (and item metadata) for one category."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    targets = [("reviews", f"raw/review_categories/{domain}.jsonl")]
    if with_meta:
        targets.append(("meta", f"raw/meta_categories/meta_{domain}.jsonl"))

    for kind, remote in targets:
        print(f"  {domain:24} {kind:8} ...", end=" ", flush=True)
        local = hf_hub_download(
            repo_id=HF_REPO,
            filename=remote,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
        size_mb = __import__("os").path.getsize(local) / 1048576
        print(f"{size_mb:8.1f} MB")
        paths[kind] = local

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="two small domains only")
    ap.add_argument("--domains", nargs="+", help="explicit category names")
    ap.add_argument("--no-meta", action="store_true", help="skip item metadata")
    args = ap.parse_args()

    domains = args.domains or (DOMAINS_SMOKE if args.smoke else DOMAINS)
    print(f"Downloading {len(domains)} domain(s) into {RAW_DIR}\n")

    for d in domains:
        download_domain(d, with_meta=not args.no_meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
