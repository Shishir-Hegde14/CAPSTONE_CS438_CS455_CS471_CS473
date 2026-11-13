#!/usr/bin/env python3
"""Local, single-purpose outbound guard for free hosted or sandbox-local OS reproduction.

The experiment runner talks only to this process on 127.0.0.1. For hosted
profiles the real provider token is read once from GUARD_KEY_ENV (e.g.
CLOUDFLARE_API_TOKEN), removed from os.environ, and never written to logs,
manifests, results, or the zip. For the sandbox-local HF profile, GUARD_KEY_ENV
names a non-secret loopback token and the only upstream is HTTP 127.0.0.1 on an
explicit port. Exactly one upstream host/path/model is reachable in either case.

The runner sees only GUARD_CLIENT_TOKEN, a random local token that is NOT the
provider key. Every request is budget-checked and append-only logged. For the
default Cloudflare Workers Free path the guard also enforces a conservative
Neurons-per-day cap so the run stops before the provider's free allocation is
exhausted; an exhausted Workers Free allocation returns 429 rather than billing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

ALLOWED_REQUEST_KEYS = {"model", "messages", "temperature", "seed", "max_tokens", "stream"}
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_day() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Guard:
    def __init__(self, testing: bool = False, **overrides: Any) -> None:
        cfg: dict[str, Any] = {
            "key_env": os.environ.get("GUARD_KEY_ENV", "CLOUDFLARE_API_TOKEN"),
            "client_token": os.environ.get("GUARD_CLIENT_TOKEN", ""),
            "upstream_base": os.environ.get(
                "GUARD_UPSTREAM_BASE",
                "https://api.cloudflare.com/client/v4/accounts/test/ai/v1",
            ).rstrip("/"),
            "allowed_hosts": os.environ.get("GUARD_ALLOWED_HOSTS", "api.cloudflare.com"),
            "allowed_path": os.environ.get("GUARD_ALLOWED_PATH", ""),
            "allowed_model": os.environ.get(
                "GUARD_ALLOWED_MODEL", "@cf/meta/llama-3.1-8b-instruct-fp8"
            ),
            "max_requests": env_int("GUARD_MAX_REQUESTS", 1600),
            "max_output_tokens": env_int("GUARD_MAX_OUTPUT_TOKENS", 512),
            "max_total_output_tokens": env_int("GUARD_MAX_TOTAL_OUTPUT_TOKENS", 250000),
            "max_total_tokens": env_int("GUARD_MAX_TOTAL_TOKENS", 1000000),
            "max_usd": env_float("GUARD_MAX_USD", 0.0),
            "input_usd_per_mtok": env_float("GUARD_INPUT_USD_PER_MTOK", 0.0),
            "output_usd_per_mtok": env_float("GUARD_OUTPUT_USD_PER_MTOK", 0.0),
            "max_neurons": env_float("GUARD_MAX_NEURONS", 9000.0),
            "input_neurons_per_mtok": env_float("GUARD_INPUT_NEURONS_PER_MTOK", 13778.0),
            "output_neurons_per_mtok": env_float("GUARD_OUTPUT_NEURONS_PER_MTOK", 26128.0),
            "max_prompt_chars": env_int("GUARD_MAX_PROMPT_CHARS", 65536),
            "max_messages": env_int("GUARD_MAX_MESSAGES", 32),
            "timeout": env_int("GUARD_TIMEOUT", 600),
            "strip_seed": os.environ.get("GUARD_STRIP_SEED", "1") == "1",
            "allow_grammar": os.environ.get("GUARD_ALLOW_GRAMMAR", "0") == "1",
            "max_grammar_chars": env_int("GUARD_MAX_GRAMMAR_CHARS", 16384),
            "log_dir": os.environ.get("GUARD_LOG_DIR", "results/api_guard"),
        }
        cfg.update(overrides)
        self.testing = testing
        self.cfg = cfg
        self.key_env = str(cfg["key_env"])
        self.allowed_model = str(cfg["allowed_model"])
        self.upstream_base = str(cfg["upstream_base"])
        parsed = urlparse.urlparse(self.upstream_base)
        allowed_hosts = {h.strip() for h in str(cfg["allowed_hosts"]).split(",") if h.strip()}
        self.allowed_path = str(cfg["allowed_path"] or parsed.path or "/")
        # Local HF inference uses a loopback-only HTTP server. This is deliberately
        # narrower than "allow any http": hostname must be exactly 127.0.0.1,
        # an explicit port is required, and the netloc/path must still be allowlisted.
        local_loopback = (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.port is not None
        )
        self.local_loopback = local_loopback
        # GBNF grammar is a llama.cpp-specific request key. Permit it only for
        # the sandbox-local loopback upstream, never for hosted HTTPS profiles.
        self.allow_grammar = bool(cfg["allow_grammar"]) and local_loopback
        self.max_grammar_chars = int(cfg["max_grammar_chars"])
        if (
            (parsed.scheme != "https" and not local_loopback)
            or parsed.netloc not in allowed_hosts
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
            or (not local_loopback and parsed.port not in (None, 443))
            or (local_loopback and parsed.port is None)
        ):
            raise SystemExit(
                f"Refusing upstream base {self.upstream_base!r}; allowed hosts: {sorted(allowed_hosts)}"
            )
        if parsed.path != self.allowed_path or ".." in self.allowed_path or "//" in self.allowed_path:
            raise SystemExit(
                f"Refusing upstream path {parsed.path!r}; expected exactly {self.allowed_path!r}"
            )
        self.upstream_chat = self.upstream_base + "/chat/completions"
        self.client_token = str(cfg["client_token"])
        self.max_requests = int(cfg["max_requests"])
        self.max_output_tokens = int(cfg["max_output_tokens"])
        self.max_total_output_tokens = int(cfg["max_total_output_tokens"])
        self.max_total_tokens = int(cfg["max_total_tokens"])
        self.max_usd = float(cfg["max_usd"])
        self.input_usd_per_mtok = float(cfg["input_usd_per_mtok"])
        self.output_usd_per_mtok = float(cfg["output_usd_per_mtok"])
        self.max_neurons = float(cfg["max_neurons"])
        self.input_neurons_per_mtok = float(cfg["input_neurons_per_mtok"])
        self.output_neurons_per_mtok = float(cfg["output_neurons_per_mtok"])
        self.max_prompt_chars = int(cfg["max_prompt_chars"])
        self.max_messages = int(cfg["max_messages"])
        self.timeout = int(cfg["timeout"])
        self.strip_seed = bool(cfg["strip_seed"])
        self.log_dir = str(cfg["log_dir"])
        self.budget_path = os.path.join(self.log_dir, "budget.json")
        self.log_path = os.path.join(self.log_dir, "requests.jsonl")
        self.lock = threading.Lock()

        if testing:
            # No provider key, no client token from the environment, no disk writes.
            self.key = "TEST_ONLY"
            self.client_token = self.client_token or "local-test-token"
            self.state: dict[str, Any] = self._fresh_state()
            return

        # Copy the real key into process memory, then remove it from the environment.
        self.key = os.environ.pop(self.key_env, "")
        self.client_token = os.environ.pop("GUARD_CLIENT_TOKEN", "") or self.client_token
        if not self.key:
            raise SystemExit(f"Missing {self.key_env}. Set it only in the guard process environment.")
        if not self.client_token:
            raise SystemExit("Missing GUARD_CLIENT_TOKEN. The runner must not see the provider key.")
        os.makedirs(self.log_dir, exist_ok=True)
        self.state = self._load_state()

    def _fresh_state(self) -> dict[str, Any]:
        return {
            "started_utc": utc(),
            "last_utc": None,
            "day_utc": utc_day(),
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_usd": 0.0,
            "neurons": 0.0,
            "key_env": self.key_env,
            "allowed_model": self.allowed_model,
            "allowed_path": self.allowed_path,
            "max_requests": self.max_requests,
            "max_total_output_tokens": self.max_total_output_tokens,
            "max_usd": self.max_usd,
            "max_neurons": self.max_neurons,
        }

    def _roll_neuron_day(self) -> bool:
        """Free-tier Neuron budgets are daily (00:00 UTC); other caps stay cumulative."""
        today = utc_day()
        if self.state.get("day_utc") == today:
            return False
        self.state["day_utc"] = today
        self.state["neurons"] = 0.0
        return True

    def _load_state(self) -> dict[str, Any]:
        if os.path.exists(self.budget_path):
            try:
                with open(self.budget_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for key, value in self._fresh_state().items():
                    state.setdefault(key, value)
                state["max_neurons"] = self.max_neurons
                state["allowed_path"] = self.allowed_path
                state["allowed_model"] = self.allowed_model
                state["key_env"] = self.key_env
                self.state = state
                if self._roll_neuron_day():
                    self._write_state()
                return state
            except (OSError, ValueError):
                pass
        return self._fresh_state()

    def _write_state(self) -> None:
        if self.testing:
            return
        tmp = self.budget_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.budget_path)

    def _append_log(self, record: dict[str, Any]) -> None:
        if self.testing:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()

    @staticmethod
    def _prompt_chars(messages: list[dict[str, str]]) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages)

    def check_auth(self, header: str | None) -> bool:
        return header == f"Bearer {self.client_token}"

    def validate(self, payload: Any) -> tuple[bool, str, dict[str, Any]]:
        if not isinstance(payload, dict):
            return False, "body must be a JSON object", {}
        allowed_keys = set(ALLOWED_REQUEST_KEYS)
        if self.allow_grammar:
            allowed_keys.add("grammar")
        unknown = sorted(set(payload) - allowed_keys)
        if unknown:
            return False, f"disallowed request keys: {unknown}", {}
        grammar = payload.get("grammar")
        if "grammar" in payload:
            if not self.allow_grammar:
                return False, "grammar is allowed only for the sandbox-local loopback upstream", {}
            if not isinstance(grammar, str) or not grammar.strip():
                return False, "grammar must be a non-empty string", {}
            if len(grammar) > self.max_grammar_chars:
                return False, f"grammar exceeds {self.max_grammar_chars} characters", {}
        if payload.get("model") != self.allowed_model:
            return False, f"model must be exactly {self.allowed_model!r}", {}
        messages = payload.get("messages")
        if not isinstance(messages, list) or not (1 <= len(messages) <= self.max_messages):
            return False, f"messages must be a list of 1..{self.max_messages} items", {}
        clean_messages: list[dict[str, str]] = []
        total_chars = 0
        for m in messages:
            if not isinstance(m, dict) or set(m) - {"role", "content"}:
                return False, "each message may contain only role and content", {}
            role = m.get("role")
            content = m.get("content")
            if role not in ALLOWED_ROLES or not isinstance(content, str):
                return False, "message role/content invalid; multimodal content is blocked", {}
            total_chars += len(content)
            clean_messages.append({"role": role, "content": content})
        if total_chars > self.max_prompt_chars:
            return False, f"prompt exceeds {self.max_prompt_chars} characters", {}
        if "temperature" in payload and payload["temperature"] not in (0, 0.0):
            return False, "temperature must be 0", {}
        if "seed" in payload and payload["seed"] not in (0, 0):
            return False, "seed must be 0 when supplied", {}
        if "stream" in payload and payload["stream"] is not False:
            return False, "streaming is blocked", {}
        max_tokens = payload.get("max_tokens", self.max_output_tokens)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not (1 <= max_tokens <= self.max_output_tokens):
            return False, f"max_tokens must be an integer 1..{self.max_output_tokens}", {}
        clean: dict[str, Any] = {
            "model": self.allowed_model,
            "messages": clean_messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if "seed" in payload:
            clean["seed"] = 0
        if "grammar" in payload:
            clean["grammar"] = grammar
        return True, "allowed", clean

    def _worst_tokens(self, clean: dict[str, Any]) -> tuple[int, int]:
        prompt_chars = self._prompt_chars(clean["messages"])
        prompt_tokens = max(1, (prompt_chars + 3) // 4)
        completion_tokens = int(clean["max_tokens"])
        return prompt_tokens, completion_tokens

    def _worst_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0) * self.input_usd_per_mtok + (
            completion_tokens / 1_000_000.0
        ) * self.output_usd_per_mtok

    def _neurons(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0) * self.input_neurons_per_mtok + (
            completion_tokens / 1_000_000.0
        ) * self.output_neurons_per_mtok

    def pre_budget(self, clean: dict[str, Any]) -> tuple[bool, str]:
        prompt_tokens, completion_tokens = self._worst_tokens(clean)
        usd = self._worst_usd(prompt_tokens, completion_tokens)
        neurons = self._neurons(prompt_tokens, completion_tokens)
        with self.lock:
            if self._roll_neuron_day():
                self._write_state()
            if self.state["requests"] + 1 > self.max_requests:
                return False, f"request cap {self.max_requests} reached"
            if self.state["completion_tokens"] + completion_tokens > self.max_total_output_tokens:
                return False, f"output-token cap {self.max_total_output_tokens} would be exceeded"
            if self.state["total_tokens"] + prompt_tokens + completion_tokens > self.max_total_tokens:
                return False, f"total-token cap {self.max_total_tokens} would be exceeded"
            if self.state["estimated_usd"] + usd > self.max_usd:
                return False, f"estimated-USD cap {self.max_usd:.2f} would be exceeded"
            if self.max_neurons > 0 and float(self.state.get("neurons", 0.0)) + neurons > self.max_neurons:
                return False, f"free-tier Neuron cap {self.max_neurons:.0f} would be exceeded"
        return True, "within budget"

    def post_budget(self, clean: dict[str, Any], body: dict[str, Any], status: int) -> dict[str, Any]:
        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prompt_chars = self._prompt_chars(clean["messages"])
        prompt_tokens = as_int(usage.get("prompt_tokens"), max(1, (prompt_chars + 3) // 4))
        completion_tokens = as_int(usage.get("completion_tokens"), int(clean["max_tokens"]))
        total_tokens = as_int(usage.get("total_tokens"), prompt_tokens + completion_tokens)
        usd = self._worst_usd(prompt_tokens, completion_tokens)
        neurons = self._neurons(prompt_tokens, completion_tokens)
        with self.lock:
            self._roll_neuron_day()
            self.state["requests"] += 1
            self.state["prompt_tokens"] += prompt_tokens
            self.state["completion_tokens"] += completion_tokens
            self.state["total_tokens"] += total_tokens
            self.state["estimated_usd"] = round(self.state["estimated_usd"] + usd, 6)
            self.state["neurons"] = round(float(self.state.get("neurons", 0.0)) + neurons, 6)
            self.state["last_utc"] = utc()
            self._write_state()
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_usd": round(usd, 6),
            "neurons": round(neurons, 6),
            "upstream_status": status,
        }

    def forward(self, clean: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.testing:
            return 200, {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        upstream = dict(clean)
        if self.strip_seed:
            upstream.pop("seed", None)
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        req = urlrequest.Request(
            self.upstream_chat,
            data=json.dumps(upstream).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as r:
                status = getattr(r, "status", 200)
                raw = r.read(2_000_001)
        except urlerror.HTTPError as exc:
            status = exc.code
            raw = exc.read(2_000_001)
        except Exception as exc:  # noqa: BLE001
            return 502, {"error": {"message": f"guard upstream error: {type(exc).__name__}"}}
        if len(raw) > 2_000_000:
            return 502, {"error": {"message": "upstream response exceeded 2 MB guard cap"}}
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
            if not isinstance(body, dict):
                raise ValueError("not an object")
        except ValueError:
            body = {"error": {"message": "upstream returned non-JSON response"}}
        return status, body

    def _log(
        self,
        decision: str,
        reason: str,
        clean: dict[str, Any] | None,
        status: int,
        accounting: dict[str, Any] | None,
    ) -> None:
        clean = clean or {}
        digest = hashlib.sha256(
            json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        record = {
            "ts_utc": utc(),
            "decision": decision,
            "reason": reason,
            "status": status,
            "model": clean.get("model", self.allowed_model),
            "prompt_chars": self._prompt_chars(clean.get("messages", [])),
            "message_count": len(clean.get("messages", [])),
            "max_tokens": clean.get("max_tokens"),
            "request_sha256_16": digest,
            "accounting": accounting,
            "cumulative": dict(self.state),
        }
        self._append_log(record)

    def process_chat(self, payload: Any, auth_header: str | None) -> tuple[int, dict[str, Any]]:
        if not self.check_auth(auth_header):
            self._log("rejected", "invalid local client token", None, 401, None)
            return 401, {"error": {"message": "invalid local guard token"}}
        ok, reason, clean = self.validate(payload)
        if not ok:
            self._log("rejected", reason, None, 400, None)
            return 400, {"error": {"message": reason}}
        ok, reason = self.pre_budget(clean)
        if not ok:
            self._log("rejected", reason, clean, 429, None)
            return 429, {"error": {"message": reason, "budget": dict(self.state)}}
        status, body = self.forward(clean)
        accounting = self.post_budget(clean, body, status)
        self._log("allowed", "forwarded", clean, status, accounting)
        return status, body

    def health(self) -> dict[str, Any]:
        parsed = urlparse.urlparse(self.upstream_base)
        return {
            "ok": True,
            "ts_utc": utc(),
            "upstream_scheme": parsed.scheme,
            "upstream_host": parsed.netloc,
            "local_loopback": self.local_loopback,
            "allowed_path": self.allowed_path,
            "allowed_model": self.allowed_model,
            "provider_key_env": self.key_env,
            "provider_key_in_guard_environ": self.key_env in os.environ,
            "strip_seed": self.strip_seed,
            "allow_grammar": self.allow_grammar,
            "state": dict(self.state),
        }

    def static_catalog(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.allowed_model,
                    "object": "model",
                    "owned_by": "local-api-guard",
                }
            ],
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "OSApiGuard/1.0"

    @property
    def guard(self) -> Guard:
        return self.server.guard  # type: ignore[attr-defined]

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse.urlparse(self.path).path
        if path == "/health":
            self._send_json(200, self.guard.health())
            return
        if path == "/v1/models":
            if not self.guard.check_auth(self.headers.get("Authorization")):
                self._send_json(401, {"error": {"message": "invalid local guard token"}})
                return
            self._send_json(200, self.guard.static_catalog())
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse.urlparse(self.path).path
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "only /v1/chat/completions is exposed"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000:
            self._send_json(400, {"error": {"message": "invalid or oversized request body"}})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._send_json(400, {"error": {"message": "body must be JSON"}})
            return
        status, body = self.guard.process_chat(payload, self.headers.get("Authorization"))
        self._send_json(status, body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def run_self_test() -> None:
    g = Guard(
        testing=True,
        client_token="local-test-token",
        allowed_model="exact/model",
        upstream_base="https://api.cloudflare.com/client/v4/accounts/test/ai/v1",
        allowed_hosts="api.cloudflare.com",
        allowed_path="/client/v4/accounts/test/ai/v1",
        max_requests=2,
        max_output_tokens=16,
        max_total_output_tokens=20,
        max_usd=1.0,
        max_neurons=0.0,
    )
    assert g.allowed_path == "/client/v4/accounts/test/ai/v1", "unexpected allowed path"
    assert g.strip_seed is True, "seed should be stripped for free providers requiring seed >= 1"
    base = {
        "model": "exact/model",
        "messages": [{"role": "user", "content": "Reply with exactly OK"}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 8,
    }
    status, _ = g.process_chat(base, "Bearer local-test-token")
    assert status == 200, "first allowed call failed"
    for bad in (
        dict(base, model="other/model"),
        dict(base, max_tokens=17),
        dict(base, stream=True),
        dict(base, tools=[]),
        dict(base, n=2),
        dict(base, grammar='root ::= "OK"'),
    ):
        status, _ = g.process_chat(bad, "Bearer local-test-token")
        assert status == 400, f"should have rejected {bad}"
    status, _ = g.process_chat(base, "Bearer wrong")
    assert status == 401, "bad local token should be rejected"
    status, _ = g.process_chat(base, "Bearer local-test-token")
    assert status == 200, "second allowed call failed"
    status, _ = g.process_chat(base, "Bearer local-test-token")
    assert status == 429, "request cap should have stopped the third call"
    assert g.state["requests"] == 2, g.state
    dumped = json.dumps(g.state)
    assert g.key not in dumped, "state must not contain the provider key"
    assert g.client_token not in dumped, "state must not contain the local client token"

    # Sandbox-local HF path: HTTP is permitted only for an exact loopback netloc,
    # and seed=0 is forwarded because llama.cpp accepts it.
    local = Guard(
        testing=True,
        client_token="local-test-token",
        allowed_model="local/model",
        upstream_base="http://127.0.0.1:11435/v1",
        allowed_hosts="127.0.0.1:11435",
        allowed_path="/v1",
        strip_seed=False,
        allow_grammar=True,
        max_grammar_chars=100,
        max_requests=2,
        max_output_tokens=16,
        max_total_output_tokens=40,
        max_usd=0.0,
        max_neurons=0.0,
    )
    assert local.local_loopback is True, "loopback HTTP upstream should be recognized"
    assert local.strip_seed is False, "llama.cpp accepts seed 0; do not strip it"
    assert local.allow_grammar is True, "loopback guard should permit llama.cpp GBNF grammar"
    assert local.health()["upstream_scheme"] == "http", "health should report local HTTP"
    assert local.health()["allow_grammar"] is True, "health should report grammar permission"
    local_base = {
        "model": "local/model",
        "messages": [{"role": "user", "content": "Reply with exactly OK"}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 8,
    }
    status, _ = local.process_chat(local_base, "Bearer local-test-token")
    assert status == 200, "local loopback guard should accept one call"
    ok, reason, clean = local.validate(dict(local_base, grammar='root ::= "OK"'))
    assert ok and clean.get("grammar") == 'root ::= "OK"', f"grammar should be forwarded: {reason}"
    ok, reason, _ = local.validate(dict(local_base, grammar="x" * 101))
    assert not ok and "grammar exceeds" in reason, "oversized grammar should be rejected"
    status, _ = local.process_chat(dict(local_base, grammar='root ::= "OK"'), "Bearer local-test-token")
    assert status == 200, "local loopback guard should accept grammar-constrained call"
    for bad_upstream in (
        "http://localhost:11435/v1",
        "http://example.com:11435/v1",
        "http://127.0.0.1/v1",
    ):
        try:
            Guard(
                testing=True,
                upstream_base=bad_upstream,
                allowed_hosts=urlparse.urlparse(bad_upstream).netloc,
                allowed_path="/v1",
            )
        except SystemExit:
            pass
        else:
            raise AssertionError(f"should have refused non-loopback/no-port HTTP {bad_upstream}")
    print("api_guard self-test: OK (hosted HTTPS + sandbox-local loopback + opt-in GBNF grammar)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local one-model API guard")
    ap.add_argument("--host", default=os.environ.get("GUARD_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=env_int("GUARD_PORT", 11436))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--reset-budget", action="store_true", help="Start a fresh budget; off by default so restarts stay capped")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.reset_budget:
        budget = os.path.join(os.environ.get("GUARD_LOG_DIR", "results/api_guard"), "budget.json")
        try:
            os.remove(budget)
        except FileNotFoundError:
            pass
    guard = Guard()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.guard = guard  # type: ignore[attr-defined]
    neuron_note = f", {guard.max_neurons:.0f} Neurons/day" if guard.max_neurons > 0 else ""
    print(
        f"api_guard listening on http://{args.host}:{args.port}/v1 "
        f"for {guard.allowed_model!r}; provider key removed from environ; "
        f"caps: {guard.max_requests} requests, {guard.max_total_output_tokens} output tokens, "
        f"${guard.max_usd:.2f} estimate{neuron_note}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
