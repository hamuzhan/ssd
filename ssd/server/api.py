"""FastAPI app exposing OpenAI-compatible endpoints for SSD."""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ssd.sampling_params import SamplingParams
from ssd.server.async_llm import AsyncLLM
from ssd.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatDelta,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
    ModelCard,
    ModelList,
    Usage,
)


def _normalize_prompt(prompt, tokenizer) -> list[int]:
    """OpenAI completions accepts str | list[int] | list[str] | list[list[int]].
    Only the single-prompt forms are supported here."""
    if isinstance(prompt, str):
        return tokenizer.encode(prompt)
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise HTTPException(status_code=400, detail="empty prompt")
        if isinstance(prompt[0], int):
            return list(prompt)
        if isinstance(prompt[0], str):
            if len(prompt) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="batched string prompts are not supported; send one request per prompt",
                )
            return tokenizer.encode(prompt[0])
        if isinstance(prompt[0], list):
            if len(prompt) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="batched token-id prompts are not supported; send one request per prompt",
                )
            return list(prompt[0])
    raise HTTPException(status_code=400, detail="unsupported prompt shape")


def _build_sampling_params(
    temperature: float,
    max_tokens: int | None,
    prompt_len: int,
    max_model_len: int,
    ignore_eos: bool,
) -> SamplingParams:
    if max_tokens is None:
        max_tokens = max(1, max_model_len - prompt_len)
    # Keep within the engine's remaining context window.
    max_tokens = max(1, min(max_tokens, max_model_len - prompt_len))
    return SamplingParams(
        temperature=temperature,
        max_new_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )


def _sse(data: dict | str) -> bytes:
    if data == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")


class _IncrementalDetokenizer:
    """Decode accumulating token ids, emit the new suffix string per call.

    Mirrors bench/chat.py's decoder worker: decode the full id list each call
    and diff against the previous string so we never emit partial UTF-8.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.prev: str = ""

    def feed(self, new_ids: list[int]) -> str:
        if not new_ids:
            return ""
        self.ids.extend(new_ids)
        full = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        delta = full[len(self.prev):]
        self.prev = full
        return delta


def create_app(async_llm: AsyncLLM, served_model_name: str) -> FastAPI:
    app = FastAPI(title="SSD OpenAI-compatible server")

    @app.on_event("startup")
    async def _startup() -> None:
        await async_llm.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await async_llm.stop()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=served_model_name)])

    async def _run_stream(
        request: Request,
        token_ids: list[int],
        sp: SamplingParams,
        request_id: str,
        formatter,
        include_usage: bool,
    ) -> AsyncIterator[bytes]:
        detok = _IncrementalDetokenizer(async_llm.engine.tokenizer)
        seq_id: int | None = None
        prompt_tokens = len(token_ids)
        completion_tokens = 0
        finish_reason: str | None = None
        finished_naturally = False
        iterator = async_llm.submit(token_ids, sp, request_id)
        try:
            first = await iterator.__anext__()
            seq_id = first.seq_id
            if first.error:
                raise HTTPException(status_code=500, detail=first.error)
            opening = formatter.opening(request_id)
            if opening is not None:
                yield _sse(opening)

            async for ev in iterator:
                if await request.is_disconnected():
                    break
                if ev.error:
                    finish_reason = "error"
                    break
                text = detok.feed(ev.delta_token_ids) if ev.delta_token_ids else ""
                completion_tokens += len(ev.delta_token_ids)
                if text:
                    yield _sse(formatter.delta(request_id, text))
                if ev.finished:
                    finish_reason = ev.finish_reason or "stop"
                    finished_naturally = True
                    break

            if finished_naturally:
                yield _sse(formatter.final(request_id, finish_reason or "stop"))
                if include_usage:
                    yield _sse(formatter.usage_chunk(
                        request_id, prompt_tokens, completion_tokens,
                    ))
                yield _sse("[DONE]")
        finally:
            if seq_id is not None and not finished_naturally:
                async_llm.abort(seq_id)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw: Request):
        tokenizer = async_llm.engine.tokenizer
        token_ids = _normalize_prompt(req.prompt, tokenizer)
        sp = _build_sampling_params(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            prompt_len=len(token_ids),
            max_model_len=async_llm.engine.config.max_model_len,
            ignore_eos=req.ignore_eos,
        )
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_usage = bool(
            req.stream_options and req.stream_options.get("include_usage")
        )

        if req.stream:
            formatter = _CompletionFormatter(served_model_name, created)
            return StreamingResponse(
                _run_stream(raw, token_ids, sp, request_id, formatter, include_usage),
                media_type="text/event-stream",
            )

        text, completion_tokens, finish_reason = await _collect(
            async_llm, token_ids, sp, request_id,
        )
        return CompletionResponse(
            id=request_id,
            created=created,
            model=served_model_name,
            choices=[CompletionChoice(
                text=text, finish_reason=finish_reason,
            )],
            usage=Usage(
                prompt_tokens=len(token_ids),
                completion_tokens=completion_tokens,
                total_tokens=len(token_ids) + completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request):
        tokenizer = async_llm.engine.tokenizer
        messages = [m.model_dump() for m in req.messages]
        try:
            token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"apply_chat_template failed: {e}")
        sp = _build_sampling_params(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            prompt_len=len(token_ids),
            max_model_len=async_llm.engine.config.max_model_len,
            ignore_eos=req.ignore_eos,
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_usage = bool(
            req.stream_options and req.stream_options.get("include_usage")
        )

        if req.stream:
            formatter = _ChatFormatter(served_model_name, created)
            return StreamingResponse(
                _run_stream(raw, token_ids, sp, request_id, formatter, include_usage),
                media_type="text/event-stream",
            )

        text, completion_tokens, finish_reason = await _collect(
            async_llm, token_ids, sp, request_id,
        )
        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=served_model_name,
            choices=[ChatCompletionChoice(
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )],
            usage=Usage(
                prompt_tokens=len(token_ids),
                completion_tokens=completion_tokens,
                total_tokens=len(token_ids) + completion_tokens,
            ),
        )

    @app.exception_handler(HTTPException)
    async def _http_exc_handler(_: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "invalid_request_error"}},
        )

    return app


async def _collect(
    async_llm: AsyncLLM,
    token_ids: list[int],
    sp: SamplingParams,
    request_id: str,
) -> tuple[str, int, str]:
    detok = _IncrementalDetokenizer(async_llm.engine.tokenizer)
    completion_tokens = 0
    finish_reason = "stop"
    async for ev in async_llm.submit(token_ids, sp, request_id):
        if ev.error:
            raise HTTPException(status_code=500, detail=ev.error)
        if ev.delta_token_ids:
            detok.feed(ev.delta_token_ids)
            completion_tokens += len(ev.delta_token_ids)
        if ev.finished:
            finish_reason = ev.finish_reason or "stop"
            break
    return detok.prev, completion_tokens, finish_reason


# ── SSE chunk formatters ──────────────────────────────────────────────────

class _CompletionFormatter:
    def __init__(self, model: str, created: int):
        self.model = model
        self.created = created

    def opening(self, request_id: str):
        return None  # completions API doesn't need a priming frame

    def _chunk(self, request_id: str, text: str, finish_reason: str | None):
        return CompletionStreamResponse(
            id=request_id,
            created=self.created,
            model=self.model,
            choices=[CompletionStreamChoice(text=text, finish_reason=finish_reason)],
        ).model_dump(exclude_none=True)

    def delta(self, request_id: str, text: str):
        return self._chunk(request_id, text, None)

    def final(self, request_id: str, finish_reason: str):
        return self._chunk(request_id, "", finish_reason)

    def usage_chunk(self, request_id: str, prompt_tokens: int, completion_tokens: int):
        return CompletionStreamResponse(
            id=request_id,
            created=self.created,
            model=self.model,
            choices=[],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        ).model_dump(exclude_none=True)


class _ChatFormatter:
    def __init__(self, model: str, created: int):
        self.model = model
        self.created = created

    def opening(self, request_id: str):
        return ChatCompletionStreamResponse(
            id=request_id,
            created=self.created,
            model=self.model,
            choices=[ChatCompletionStreamChoice(delta=ChatDelta(role="assistant", content=""))],
        ).model_dump(exclude_none=True)

    def delta(self, request_id: str, text: str):
        return ChatCompletionStreamResponse(
            id=request_id,
            created=self.created,
            model=self.model,
            choices=[ChatCompletionStreamChoice(delta=ChatDelta(content=text))],
        ).model_dump(exclude_none=True)

    def final(self, request_id: str, finish_reason: str):
        return ChatCompletionStreamResponse(
            id=request_id,
            created=self.created,
            model=self.model,
            choices=[ChatCompletionStreamChoice(delta=ChatDelta(), finish_reason=finish_reason)],
        ).model_dump(exclude_none=True)

    def usage_chunk(self, request_id: str, prompt_tokens: int, completion_tokens: int):
        return ChatCompletionStreamResponse(
            id=request_id,
            created=self.created,
            model=self.model,
            choices=[],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        ).model_dump(exclude_none=True)
