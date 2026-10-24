"""Deterministic stand-ins for real models, used by tests and the offline demo.

``OfflineProvider`` does *extractive* answering: it parses the grounded-answer prompt built by
:mod:`gitcontext.answer.prompts`, scores sentences from each numbered source against the question
and stitches the best ones together with ``[n]`` citations. It never invents text that is not in the
sources, which keeps the offline demo honest, but it is not a substitute for a real model's
synthesis quality.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from .base import LLMResponse, Message, estimate_tokens

Handler = Callable[[list[Message], str | None, dict[str, Any] | None], str]


class ScriptedLLM:
    """Returns pre-programmed responses in order (or via a callable). Records every call."""

    model = "scripted"

    def __init__(self, responses: list[str] | Handler):
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system, "json_schema": json_schema})
        if callable(self._responses):
            text = self._responses(messages, system, json_schema)
        else:
            if not self._responses:
                raise RuntimeError("ScriptedLLM ran out of responses")
            text = self._responses.pop(0)
        prompt = (system or "") + "".join(m.content for m in messages)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            model=self.model,
        )


_SOURCE_RE = re.compile(
    r'<source id="(?P<id>\d+)" type="(?P<type>[a-z]+)" title="(?P<title>[^"]*)"'
    r' location="(?P<loc>[^"]*)">\n(?P<body>.*?)\n</source>',
    re.DOTALL,
)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """a an and are as at be by do does did for from how i in is it its of on or our so that the
    this to was we what when where which who why with you your there their they them about work
    works can should would could have has had been were will""".split()  # noqa: SIM905
)
REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"


def _terms(text: str) -> set[str]:
    out: set[str] = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.add(w)
        if len(w) > 4 and w.endswith("s"):
            out.add(w[:-1])
        for suffix in ("ing", "ed", "es"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                out.add(w[: -len(suffix)])
    return out


def _sentences(source_type: str, body: str) -> list[str]:
    if source_type == "code":
        # Use docstrings and comments: the human-language part of code.
        found: list[str] = []
        for doc in re.findall(r'"""(.*?)"""', body, re.DOTALL):
            found.extend(_split_prose(doc))
        for comment in re.findall(r"(?:#|//)\s?(.+)", body):
            if len(comment.split()) >= 4:
                found.append(comment.strip())
        return found
    out: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)", body):
        block = block.strip()
        if not block or block.startswith(("#", "|", "---", "Files changed", "Files:")):
            continue
        out.extend(_split_prose(block))
    return out


def _split_prose(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:[-*]|\d+\.)\s+", "", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`*\[(])", text)
    return [p.strip(" -*") for p in parts if 5 <= len(p.split()) <= 70]


_META_RE = re.compile(r"^\((?:author: (?P<author>.*?))?(?:, )?(?:date: (?P<date>[^)]*))?\)\n")


def extractive_answer(messages: list[Message], max_sentences: int = 4) -> str:
    """Pick the best-matching sentence per source and stitch them together with citations."""
    prompt = "\n".join(m.content for m in messages)
    q_match = _QUESTION_RE.search(prompt)
    question = q_match.group(1) if q_match else prompt[-500:]
    q_terms = _terms(question)
    if not q_terms:
        return REFUSAL_TOKEN
    picks: list[tuple[float, int, str]] = []
    for m in _SOURCE_RE.finditer(prompt):
        sid, stype, title, loc = int(m["id"]), m["type"], m["title"], m["loc"]
        body = m["body"]
        meta = _META_RE.match(body)
        if meta:
            body = body[meta.end() :]
        title_terms = _terms(title)
        best: tuple[float, str] | None = None
        for sent in _sentences(stype, body):
            if sent.rstrip(".") in title:
                continue  # the title is already in the lead
            s_terms = _terms(sent)
            hits = len(q_terms & s_terms) + 0.5 * len(q_terms & (title_terms - s_terms))
            overlap = hits / len(q_terms)
            if sent.rstrip().endswith("?"):
                overlap *= 0.5  # a reviewer's question restates the query; it is not an answer
            if best is None or overlap > best[0]:
                best = (overlap, sent)
        if best and best[0] >= 0.3:
            score = best[0] * (1 - 0.1 * (sid - 1))  # trust the retriever's ranking
            lead = _lead(
                stype, title, loc, meta["author"] if meta else None, meta["date"] if meta else None
            )
            picks.append((score, sid, f"{lead}{best[1].rstrip('.')} [{sid}]."))
    if not picks:
        return REFUSAL_TOKEN
    top = max(p[0] for p in picks)
    picks = [p for p in picks if p[0] >= 0.6 * top]
    picks.sort(key=lambda p: p[0], reverse=True)
    chosen = sorted(picks[:max_sentences], key=lambda p: p[1])
    return " ".join(text for _, _, text in chosen)


def _lead(stype: str, title: str, loc: str, author: str | None, date: str | None) -> str:
    if stype == "code":
        return f"`{title}` (`{loc}`): "
    if stype == "pr":
        by = f" by {author}" if author else ""
        return f"{title}{by}: "
    if stype == "incident":
        return f"The postmortem {title.split(' > ')[0]} notes: "
    if stype == "commit":
        who = f" by {author}" if author else ""
        when = f" on {date}" if date else ""
        return f"Commit {loc}{who}{when}: "
    if stype == "discussion":
        return f"{title}: "
    return f"{title} ({loc}): "


class OfflineProvider:
    """Deterministic, network-free provider. Default behaviour is extractive grounded answering."""

    model = "offline-extractive"

    def __init__(self, handler: Handler | None = None):
        self._handler = handler

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        if self._handler is not None:
            text = self._handler(messages, system, json_schema)
        else:
            text = extractive_answer(messages)
        prompt = (system or "") + "".join(m.content for m in messages)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
        )
