#!/usr/bin/env python3
"""Deterministic OpenAI-compatible mock server for plumbing/timeout tests.

It returns the published ground-truth OS script for the current instruction and
then `Act: finish`.  It is NOT an LLM and must never be used to report model
accuracy.  Use it to verify replay prompts, response parsing, checkpointing and
the separation between model and shell timeouts without a GPU.

Usage:
  python tools/mock_model_server.py --data data/os_tasks.json --port 8765 \
      [--delay 0.2]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class Handler(BaseHTTPRequestHandler):
    task_by_instruction: dict[str, dict[str, Any]] = {}
    delay: float = 0.0

    def log_message(self, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = payload.get("messages", [])
        instruction = None
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content") in self.task_by_instruction:
                instruction = message["content"]
                break
        if instruction is None:
            instruction = next(
                (
                    message.get("content", "")
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
        task = self.task_by_instruction.get(instruction)
        has_action = any(
            message.get("role") == "assistant" and "Act: bash" in message.get("content", "")
            for message in messages
        )
        if task is None:
            text = "Think: no matching task. Act: finish"
        elif has_action:
            text = "Think: Hello world. Act: finish"
        else:
            script = task["evaluation_info"]["ground_truth_command_item"]["script"]
            text = f"Think: Hello world. Act: bash\n```bash\n{script}\n```"

        if self.delay:
            time.sleep(self.delay)
        response = {
            "id": "mock-os-response",
            "object": "chat.completion",
            "model": payload.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages) // 4,
                "completion_tokens": len(text) // 4,
                "total_tokens": (sum(len(m.get("content", "")) for m in messages) + len(text)) // 4,
            },
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Expected when a client's model-timeout fires before the reply.
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/os_tasks.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    tasks = json.loads(Path(args.data).read_text(encoding="utf-8"))["tasks"]
    Handler.task_by_instruction = {task["instruction"]: task for task in tasks}
    Handler.delay = args.delay
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock model server listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
