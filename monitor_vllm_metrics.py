#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import time
from urllib.request import urlopen


METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")


def scrape_metrics(url: str) -> dict[str, float]:
    with urlopen(url, timeout=5) as response:
        text = response.read().decode("utf-8", errors="replace")

    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def metric(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def gpu_line(gpu_id: str | None) -> str:
    if gpu_id is None:
        return ""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except Exception:
        return " gpu=NA"
    if not output:
        return " gpu=NA"
    gpu_util, mem_util, mem_used, mem_total, power = [part.strip() for part in output.split(",")]
    return f" gpu={gpu_util}% mem={mem_used}/{mem_total}MiB mem_util={mem_util}% power={power}W"


def rate(curr: dict[str, float], prev: dict[str, float], name: str, elapsed: float) -> float:
    return max(0.0, metric(curr, name) - metric(prev, name)) / elapsed if elapsed > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor vLLM Prometheus metrics.")
    parser.add_argument("--url", default="http://127.0.0.1:8006/metrics")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--gpu-id", default=None)
    args = parser.parse_args()

    prev = scrape_metrics(args.url)
    prev_time = time.time()
    print(
        "time running waiting kv% prompt_tok/s gen_tok/s total_tok/s req/s "
        "prefix_hit% preemptions gpu",
        flush=True,
    )

    while True:
        time.sleep(args.interval)
        curr = scrape_metrics(args.url)
        now = time.time()
        elapsed = now - prev_time

        prompt_tps = rate(curr, prev, "vllm:prompt_tokens_total", elapsed)
        gen_tps = rate(curr, prev, "vllm:generation_tokens_total", elapsed)
        req_s = rate(curr, prev, "vllm:request_success_total", elapsed)
        prefix_queries = max(0.0, metric(curr, "vllm:prefix_cache_queries_total") - metric(prev, "vllm:prefix_cache_queries_total"))
        prefix_hits = max(0.0, metric(curr, "vllm:prefix_cache_hits_total") - metric(prev, "vllm:prefix_cache_hits_total"))
        prefix_hit_pct = (prefix_hits / prefix_queries * 100.0) if prefix_queries else 0.0
        print(
            f"{time.strftime('%H:%M:%S')} "
            f"running={metric(curr, 'vllm:num_requests_running'):.0f} "
            f"waiting={metric(curr, 'vllm:num_requests_waiting'):.0f} "
            f"kv={metric(curr, 'vllm:kv_cache_usage_perc') * 100:.1f}% "
            f"prompt={prompt_tps:.1f} "
            f"gen={gen_tps:.1f} "
            f"total={prompt_tps + gen_tps:.1f} "
            f"req={req_s:.2f} "
            f"prefix_hit={prefix_hit_pct:.1f}% "
            f"preemptions={metric(curr, 'vllm:num_preemptions_total'):.0f}"
            f"{gpu_line(args.gpu_id)}",
            flush=True,
        )

        prev = curr
        prev_time = now


if __name__ == "__main__":
    main()
