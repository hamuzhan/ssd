"""SSD OpenAI-compatible server launcher.

Usage:
    python -O -m ssd.server --model /path/to/target --port 40030

For speculative decoding:
    python -O -m ssd.server --model $LLAMA_70B --draft $LLAMA_1B \
        --tensor-parallel-size 4 --speculate --speculate-k 6 --port 40030

For async SSD (draft on its own GPU; --tensor-parallel-size counts target GPUs,
total GPUs used = --tensor-parallel-size + 1):
    python -O -m ssd.server --model $LLAMA_70B --draft $LLAMA_1B \
        --tensor-parallel-size 4 --speculate --draft-async \
        --speculate-k 7 --async-fan-out 3 --port 40030
"""
from __future__ import annotations

import ssd.paths  # noqa: F401 — sets TORCH_CUDA_ARCH_LIST before flashinfer imports

import argparse
import sys

from ssd.engine.llm_engine import LLMEngine
from ssd.server.api import create_app
from ssd.server.async_llm import AsyncLLM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ssd.server", description=__doc__)

    # Model
    p.add_argument("--model", required=True, help="Target model directory (HF snapshot path)")
    p.add_argument("--draft", default=None, help="Draft model directory (required for --speculate)")
    p.add_argument("--served-model-name", default=None,
                   help="Name reported by /v1/models (defaults to --model basename)")

    # Parallelism
    p.add_argument("--tensor-parallel-size", "--tp", type=int, default=1, dest="tp_size",
                   help="Target tensor-parallel size. Total GPUs = tp_size (+1 if --draft-async).")

    # Speculation
    p.add_argument("--speculate", action="store_true")
    p.add_argument("--speculate-k", type=int, default=1)
    p.add_argument("--draft-async", action="store_true",
                   help="Async SSD mode (draft on a dedicated GPU)")
    p.add_argument("--async-fan-out", type=int, default=3)
    p.add_argument("--sampler-x", type=float, default=None)
    p.add_argument("--jit-speculate", action=argparse.BooleanOptionalAction, default=True,
                   help="On cache miss, run real draft forward (jit) instead of random tokens")
    p.add_argument("--use-eagle", action="store_true")

    # Memory / batching
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--kvcache-block-size", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--enforce-eager", action="store_true")

    # Server
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=40030)
    p.add_argument("--log-level", default="warning")

    # Misc
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)
    if args.speculate and not args.draft:
        p.error("--speculate requires --draft")
    return args


def build_engine(args: argparse.Namespace) -> LLMEngine:
    num_gpus = args.tp_size + (1 if args.draft_async else 0)
    kwargs = dict(
        num_gpus=num_gpus,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.kvcache_block_size,
        enforce_eager=args.enforce_eager,
        speculate=args.speculate,
        speculate_k=args.speculate_k,
        draft_async=args.draft_async,
        async_fan_out=args.async_fan_out,
        sampler_x=args.sampler_x,
        jit_speculate=args.jit_speculate,
        use_eagle=args.use_eagle,
        verbose=args.verbose,
    )
    if args.draft:
        kwargs["draft"] = args.draft
    return LLMEngine(args.model, **kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn not installed. Run `uv sync --extra server` first.",
            file=sys.stderr,
        )
        return 1

    engine = build_engine(args)
    async_llm = AsyncLLM(engine)
    served_name = args.served_model_name or args.model.rstrip("/").split("/")[-1]
    app = create_app(async_llm, served_name)

    print(
        f"[ssd.server] listening on http://{args.host}:{args.port} "
        f"(model={served_name})",
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
