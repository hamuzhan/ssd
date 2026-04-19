"""Async shell around LLMEngine.

A single background task owns the engine:
  - Drains pending submissions onto the scheduler.
  - Processes abort requests coming from disconnected clients.
  - Drives engine.step() via asyncio.to_thread so the loop stays free to
    write SSE frames between ticks.

Streaming mirrors LLMEngine.generate()'s existing `stream_callback` pattern
(per-seq `_stream_lens` delta tracking) — no new accounting.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from ssd.engine.llm_engine import LLMEngine
from ssd.engine.step import InferenceStep
from ssd.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    seq_id: int
    delta_token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    error: str | None = None


@dataclass
class _SubmitRequest:
    token_ids: list[int]
    sampling_params: SamplingParams
    queue: asyncio.Queue
    request_id: str


class AsyncLLM:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._pending: asyncio.Queue[_SubmitRequest] = asyncio.Queue()
        self._abort_queue: asyncio.Queue[int] = asyncio.Queue()
        self._q_by_seq: dict[int, asyncio.Queue] = {}
        self._stream_lens: dict[int, int] = {}
        self._step: InferenceStep | None = None
        self._loop_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self) -> None:
        self._step = self.engine.create_inference_step(self.engine.config)
        self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            self.engine.exit(hard=False)
        except Exception:
            pass

    async def submit(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[StreamEvent]:
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        await self._pending.put(_SubmitRequest(
            token_ids=token_ids,
            sampling_params=sampling_params,
            queue=q,
            request_id=request_id,
        ))
        while True:
            ev = await q.get()
            if ev is None:
                return
            yield ev
            if ev.finished:
                return

    def abort(self, seq_id: int) -> None:
        try:
            self._abort_queue.put_nowait(seq_id)
        except Exception:
            pass

    async def _run(self) -> None:
        engine = self.engine
        step = self._step
        eos_id = engine.config.eos

        while not self._stopped:
            # 1. Register any pending submissions (one at a time per tick is fine;
            #    scheduler queues them into `waiting` and prefill picks them up).
            registered_any = False
            while True:
                try:
                    req = self._pending.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    seq = engine.add_request(req.token_ids, req.sampling_params)
                except Exception as e:
                    req.queue.put_nowait(StreamEvent(
                        seq_id=-1, finished=True, finish_reason="error", error=str(e),
                    ))
                    req.queue.put_nowait(None)
                    continue
                self._q_by_seq[seq.seq_id] = req.queue
                self._stream_lens[seq.seq_id] = 0
                req.queue.put_nowait(StreamEvent(
                    seq_id=seq.seq_id,
                    prompt_tokens=len(req.token_ids),
                ))
                registered_any = True

            # 2. Handle aborts from client disconnects.
            while True:
                try:
                    seq_id = self._abort_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if engine.abort(seq_id):
                    q = self._q_by_seq.pop(seq_id, None)
                    self._stream_lens.pop(seq_id, None)
                    if q is not None:
                        q.put_nowait(StreamEvent(
                            seq_id=seq_id, finished=True, finish_reason="abort",
                        ))
                        q.put_nowait(None)

            # 3. Step engine if there's work, otherwise idle.
            if engine.is_finished():
                if registered_any:
                    continue
                try:
                    req = await asyncio.wait_for(self._pending.get(), timeout=0.05)
                    self._pending.put_nowait(req)
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                outputs = await asyncio.to_thread(engine.step, step)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("engine.step crashed: %s", e)
                # Fail all in-flight requests; engine state may be corrupted.
                for sid, q in list(self._q_by_seq.items()):
                    q.put_nowait(StreamEvent(
                        seq_id=sid, finished=True, finish_reason="error", error=str(e),
                    ))
                    q.put_nowait(None)
                self._q_by_seq.clear()
                self._stream_lens.clear()
                return

            # 4. Stream deltas for running sequences (same loop as
            #    llm_engine.py's stream_callback path).
            for seq in engine.scheduler.running:
                q = self._q_by_seq.get(seq.seq_id)
                if q is None:
                    continue
                cur = seq.num_completion_tokens
                prev = self._stream_lens.get(seq.seq_id, 0)
                if cur > prev:
                    q.put_nowait(StreamEvent(
                        seq_id=seq.seq_id,
                        delta_token_ids=list(seq.completion_token_ids[prev:cur]),
                    ))
                    self._stream_lens[seq.seq_id] = cur

            # 5. Finished sequences: emit trailing delta + DONE sentinel.
            for seq_id, token_ids in outputs:
                q = self._q_by_seq.pop(seq_id, None)
                prev = self._stream_lens.pop(seq_id, 0)
                total = len(token_ids)
                tail = list(token_ids[prev:]) if total > prev else []
                finish_reason = "stop" if (token_ids and token_ids[-1] == eos_id) else "length"
                if q is not None:
                    q.put_nowait(StreamEvent(
                        seq_id=seq_id,
                        delta_token_ids=tail,
                        finished=True,
                        finish_reason=finish_reason,
                    ))
                    q.put_nowait(None)
