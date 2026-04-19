"""OpenAI-compatible request/response models.

Unsupported fields (top_p, top_k, presence_penalty, frequency_penalty, stop, ...)
are accepted for client compatibility but ignored by the engine — same behavior
as vLLM/SGLang for flags the configured backend doesn't honor.
"""
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False
    ignore_eos: bool = False
    n: int = 1
    stop: str | list[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int] | list[str] | list[list[int]]
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False
    ignore_eos: bool = False
    n: int = 1
    stop: str | list[str] | None = None
    echo: bool = False
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None
    logprobs: Any = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None
    logprobs: Any = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]
    usage: Usage | None = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None
    logprobs: Any = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChatDelta = Field(default_factory=ChatDelta)
    finish_reason: str | None = None
    logprobs: Any = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "ssd"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
