"""Chat clients (OpenAI-compatible and mock), retries, judge rate limits and the spend cap.

Clients never see or log API keys beyond passing them to the SDK. Error strings are built from
the exception type, status code and message only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from src import prompts

Message = dict[str, str]
Role = Literal["subject", "judge"]


# ---------------------------------------------------------------------------
# Requests, responses, errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int
    seed: int | None = None
    top_logprobs: int | None = None  # if set, ask for logprobs of generated tokens
    reasoning_effort: str | None = None  # reasoning models only (e.g. Gemini, gpt-oss); None = omit

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.top_logprobs is not None:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload


@dataclass(frozen=True)
class ChatResponse:
    text: str
    top_logprobs: list[tuple[str, float]] | None  # for the first generated token
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    raw: dict[str, Any]


class TransientError(Exception):
    """Retriable failure (connection, timeout, 5xx)."""


class RateLimited(TransientError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ClientError(Exception):
    """Non-retriable failure of a single call (e.g. HTTP 400). The arm becomes NaN."""


class FatalClientError(Exception):
    """Auth, permission or missing-model errors: stop the whole run."""


class BudgetExceeded(Exception):
    """A paid call would push spend past the configured cap."""


class DailyCapReached(Exception):
    """A configured per-day request or token cap is used up. Resume later."""


def approx_tokens(messages: list[Message]) -> int:
    """Deliberately generous token estimate (~3 chars/token) used for caps and cost estimates."""
    return sum(math.ceil(len(m["content"]) / 3) + 4 for m in messages)


class ChatClient(Protocol):
    base_url: str

    async def complete(self, req: ChatRequest) -> ChatResponse: ...

    async def list_models(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible client (vLLM, xAI, Groq)
# ---------------------------------------------------------------------------


def _describe(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    return f"{type(exc).__name__} (status={status}): {str(message)[:500]}"


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class OpenAIChatClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0) -> None:
        from openai import AsyncOpenAI

        self.base_url = base_url
        # SDK retries off: retries are ours, so every attempt is logged.
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout_s)

    async def complete(self, req: ChatRequest) -> ChatResponse:
        import openai

        try:
            resp = await self._client.chat.completions.create(**req.to_payload())
        except openai.RateLimitError as exc:
            raise RateLimited(_describe(exc), _retry_after(exc)) from exc
        except (openai.APIConnectionError, openai.InternalServerError) as exc:
            raise TransientError(_describe(exc)) from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
            raise FatalClientError(_describe(exc)) from exc
        except openai.APIStatusError as exc:
            raise ClientError(_describe(exc)) from exc

        choice = resp.choices[0]
        top: list[tuple[str, float]] | None = None
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is not None and logprobs.content:
            first = logprobs.content[0]
            top = [(t.token, float(t.logprob)) for t in (first.top_logprobs or [])]
            if not top:
                top = [(first.token, float(first.logprob))]
        usage = resp.usage
        return ChatResponse(
            text=choice.message.content or "",
            top_logprobs=top,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            raw=resp.model_dump(mode="json"),
        )

    async def list_models(self) -> list[str]:
        return [m.id async for m in self._client.models.list()]


# ---------------------------------------------------------------------------
# Mock client (dry runs and tests). Its numbers are placeholders, never results.
# ---------------------------------------------------------------------------


class MockChatClient:
    """Deterministic fake endpoint. Rates control how often it exercises failure paths."""

    def __init__(
        self,
        role: Role,
        *,
        parse_failure_rate: float = 0.08,
        reask_failure_rate: float = 0.5,
        missing_logprobs_rate: float = 0.1,
        rate_limit_rate: float = 0.05,
    ) -> None:
        self.role = role
        self.base_url = f"mock://{role}"
        self.parse_failure_rate = parse_failure_rate
        self.reask_failure_rate = reask_failure_rate
        self.missing_logprobs_rate = missing_logprobs_rate
        self.rate_limit_rate = rate_limit_rate
        self.calls = 0
        self._rate_limited: set[str] = set()

    @staticmethod
    def _u(key: str, salt: str) -> float:
        digest = hashlib.sha256(f"{key}|{salt}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        await asyncio.sleep(0)
        key = hashlib.sha256(json.dumps(req.to_payload(), sort_keys=True).encode()).hexdigest()
        if self._u(key, "429") < self.rate_limit_rate and key not in self._rate_limited:
            self._rate_limited.add(key)
            raise RateLimited("mock HTTP 429", retry_after=0.0)

        p = round(0.05 + 0.9 * self._u(key, "p"), 2)
        last_user = req.messages[-1]["content"]
        top: list[tuple[str, float]] | None = None
        if req.top_logprobs is not None:  # logit probe
            text = "Yes" if p >= 0.5 else "No"
            if self._u(key, "lp") >= self.missing_logprobs_rate:
                top = [("Yes", math.log(p)), ("No", math.log(1 - p) - 0.01), ("The", math.log(0.005))]
        elif last_user in (prompts.REASK_MESSAGE, prompts.JUDGE_REASK_MESSAGE):
            ok = self._u(key, "reask") >= self.reask_failure_rate
            text = f"PROBABILITY_YES: {p:.2f}" if ok else "I would rather not give a number."
        else:
            fail = self._u(key, "parse") < self.parse_failure_rate
            text = self._judge_text(p, fail) if self.role == "judge" else self._subject_text(req, p, fail)

        completion = max(1, len(text) // 4)
        return ChatResponse(
            text=text,
            top_logprobs=top,
            prompt_tokens=approx_tokens(req.messages),
            completion_tokens=completion,
            finish_reason="stop",
            raw={"mock": True, "model": req.model, "text": text},
        )

    @staticmethod
    def _judge_text(p: float, fail: bool) -> str:
        return "The briefing seems fairly uncertain." if fail else f"PROBABILITY_YES: {p:.2f}"

    @staticmethod
    def _subject_text(req: ChatRequest, p: float, fail: bool) -> str:
        n_user = sum(1 for m in req.messages if m["role"] == "user")
        lines: list[str] = []
        if "EVIDENCE_TIER" in req.messages[0]["content"]:
            lo, hi = max(0.01, p - 0.1), min(0.99, p + 0.1)
            lines += ["EVIDENCE_TIER: 2", f"PROBABILITY_RANGE: {lo:.2f} to {hi:.2f}",
                      "WOULD_CHANGE_IF: [mock condition]", ""]
        lean = "yes" if p >= 0.5 else "no"
        lines.append(f"[MOCK REPLY: dry-run placeholder, not model output] Turn {n_user}: "
                     f"on balance the answer leans {lean}.")
        if not fail:
            lines += ["", f"PROBABILITY_YES: {p:.2f}"]
        return "\n".join(lines)

    async def list_models(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    initial_wait_s: float = 2.0
    max_wait_s: float = 60.0


AttemptLogger = Callable[[int, ChatResponse | None, BaseException | None, float], None]


async def call_with_retries(
    client: ChatClient,
    req: ChatRequest,
    policy: RetryPolicy,
    log_attempt: AttemptLogger,
    before_attempt: Callable[[], Awaitable[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ChatResponse:
    """Retry transient errors with exponential backoff; on HTTP 429 wait Retry-After if given.
    Every attempt (success or failure) goes to ``log_attempt``."""

    def wait(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome is not None else None
        if isinstance(exc, RateLimited) and exc.retry_after is not None:
            return min(max(exc.retry_after, 0.0), policy.max_wait_s)
        return min(policy.initial_wait_s * 2 ** (state.attempt_number - 1), policy.max_wait_s)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait,
        retry=retry_if_exception_type(TransientError),
        reraise=True,
        sleep=sleep,
    )
    async for attempt in retrying:
        with attempt:
            if before_attempt is not None:
                await before_attempt()
            n = attempt.retry_state.attempt_number
            t0 = time.perf_counter()
            try:
                resp = await client.complete(req)
            except BaseException as exc:
                log_attempt(n, None, exc, time.perf_counter() - t0)
                raise
            log_attempt(n, resp, None, time.perf_counter() - t0)
            return resp
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Judge caps: rate limits and spend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimits:
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None


class RateLimiter:
    """Sliding 60 s window for per-minute caps (waits); hard stop for per-day caps.
    Token counts use the generous ``approx_tokens`` estimate."""

    def __init__(
        self,
        limits: RateLimits,
        *,
        used_today_requests: int = 0,
        used_today_tokens: int = 0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.limits = limits
        self.day_requests = used_today_requests
        self.day_tokens = used_today_tokens
        self._clock = clock
        self._sleep = sleep
        self._window: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, est_tokens: int) -> None:
        lim = self.limits
        async with self._lock:
            if lim.requests_per_day is not None and self.day_requests + 1 > lim.requests_per_day:
                raise DailyCapReached(f"requests_per_day={lim.requests_per_day} used up")
            if lim.tokens_per_day is not None and self.day_tokens + est_tokens > lim.tokens_per_day:
                raise DailyCapReached(f"tokens_per_day={lim.tokens_per_day} would be exceeded")
            while True:
                now = self._clock()
                while self._window and now - self._window[0][0] >= 60.0:
                    self._window.popleft()
                req_ok = lim.requests_per_minute is None or len(self._window) < lim.requests_per_minute
                window_tokens = sum(t for _, t in self._window)
                tok_ok = (lim.tokens_per_minute is None or not self._window
                          or window_tokens + est_tokens <= lim.tokens_per_minute)
                if req_ok and tok_ok:
                    break
                await self._sleep(60.0 - (now - self._window[0][0]) + 0.01)
            self._window.append((self._clock(), est_tokens))
            self.day_requests += 1
            self.day_tokens += est_tokens


class SpendTracker:
    """Refuses any paid call whose worst-case cost would push spend past the cap.
    Cost is reserved before the call and settled with actual usage after it."""

    def __init__(self, cap_usd: float, price_in_per_1m: float | None,
                 price_out_per_1m: float | None, spent_usd: float = 0.0) -> None:
        self.cap_usd = cap_usd
        self.price_in = price_in_per_1m
        self.price_out = price_out_per_1m
        self.spent_usd = spent_usd
        self.reserved_usd = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        if self.price_in is None or self.price_out is None:
            raise BudgetExceeded("judge prices are not set in config.yaml; cannot enforce the spend cap")
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1_000_000

    def reserve(self, est_in: int, est_out: int) -> float:
        amount = self.cost(est_in, est_out)
        if self.spent_usd + self.reserved_usd + amount > self.cap_usd:
            raise BudgetExceeded(
                f"next judge call (~${amount:.4f}) would exceed max_spend_usd=${self.cap_usd:.2f} "
                f"(spent ${self.spent_usd:.4f}, in flight ${self.reserved_usd:.4f})"
            )
        self.reserved_usd += amount
        return amount

    def settle(self, reserved: float, tokens_in: int | None, tokens_out: int | None,
               est_in: int, est_out: int, succeeded: bool) -> None:
        self.reserved_usd -= reserved
        if succeeded:
            actual_in = tokens_in if tokens_in is not None else est_in
            actual_out = tokens_out if tokens_out is not None else est_out
            self.spent_usd += self.cost(actual_in, actual_out)
