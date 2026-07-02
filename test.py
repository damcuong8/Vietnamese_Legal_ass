#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import sys
import time


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill GPU VRAM and keep only a target amount of free memory."
    )
    parser.add_argument("--gpu", type=int, default=1, help="Physical GPU id to fill.")
    parser.add_argument(
        "--target-free-mib",
        type=int,
        default=2,
        help="Stop when free VRAM is at or below this value.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=512,
        help="Initial allocation chunk size.",
    )
    parser.add_argument(
        "--min-chunk-mib",
        type=int,
        default=1,
        help="Smallest allocation chunk size after OOM backoff.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Hold memory for N seconds. Default 0 means hold until Ctrl+C.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_free_mib < 1:
        raise ValueError("--target-free-mib must be >= 1")
    if args.chunk_mib < args.min_chunk_mib:
        raise ValueError("--chunk-mib must be >= --min-chunk-mib")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    blocks: list[torch.Tensor] = []
    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def free_mib() -> int:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes // MIB)

    def total_mib() -> int:
        _, total_bytes = torch.cuda.mem_get_info(device)
        return int(total_bytes // MIB)

    try:
        torch.empty(1, device=device).fill_(0)
        print(
            f"GPU physical id={args.gpu}, visible cuda:0, total={total_mib()} MiB, "
            f"initial_free={free_mib()} MiB"
        )
        print(f"Target free VRAM: {args.target_free_mib} MiB")

        while not stop:
            current_free = free_mib()
            remaining_to_fill = current_free - args.target_free_mib
            if remaining_to_fill <= 0:
                break

            request_mib = min(args.chunk_mib, remaining_to_fill)
            request_mib = max(args.min_chunk_mib, request_mib)

            allocated = False
            while request_mib >= args.min_chunk_mib and not allocated:
                if free_mib() - request_mib < args.target_free_mib:
                    request_mib = max(args.min_chunk_mib, free_mib() - args.target_free_mib)
                    if request_mib < args.min_chunk_mib:
                        break

                try:
                    block = torch.empty(request_mib * MIB, dtype=torch.uint8, device=device)
                    block.fill_(0)
                    blocks.append(block)
                    allocated = True
                    print(
                        f"allocated={request_mib:>5} MiB | "
                        f"free={free_mib():>6} MiB | blocks={len(blocks)}",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    request_mib //= 2

            if not allocated:
                break

        print(f"Done. Current free VRAM: {free_mib()} MiB")
        if args.seconds > 0:
            deadline = time.monotonic() + args.seconds
            while not stop and time.monotonic() < deadline:
                time.sleep(1)
        else:
            print("Holding VRAM. Press Ctrl+C to release.")
            while not stop:
                time.sleep(1)
    finally:
        blocks.clear()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("Released allocated VRAM.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
