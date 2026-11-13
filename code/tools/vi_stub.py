#!/usr/bin/python3
"""Tiny, deterministic stand-in for the `vi` calls used by the OS benchmark.

The official image includes an interactive editor.  A no-Docker harness cannot
reasonably drive a real TTY, so this stub implements only the ex/normal commands
that appear in the published ground-truth scripts: insert blocks from stdin,
`normal i...`, `normal o...`, `call append(...)`, simple substitutions and `:wq`.
It is deliberately not a general-purpose editor.
"""

from __future__ import annotations

import os
import re
import sys


def clean(text: str) -> str:
    # Bash writes either a real ESC or one of its escaped spellings.
    text = text.replace("\x1b", "\n")
    text = re.sub(r"\\(?:x1[bB]|033|e)", "\n", text)
    return text.replace("\r", "")


def parse_args(argv: list[str]) -> tuple[str | None, list[str]]:
    target = None
    commands: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-c":
            if i + 1 < len(argv):
                commands.append(argv[i + 1])
            i += 2
        elif arg.startswith("+"):
            commands.append(arg[1:])
            i += 1
        elif arg in {"-e", "-es", "-E", "-s", "-u", "-N"}:
            i += 1
        elif arg.startswith("-"):
            i += 1
        else:
            target = arg
            i += 1
    return target, commands


def parse_stdin(raw: str) -> tuple[list[str], list[tuple[str, str, bool]]]:
    """Return inserted lines and (pattern, replacement, global) substitutions."""
    lines = clean(raw).split("\n")
    inserted: list[str] = []
    substitutions: list[tuple[str, str, bool]] = []
    inserting = False

    for line in lines:
        stripped = line.strip()
        ex = stripped[1:] if stripped.startswith(":") else stripped
        if ex in {"wq", "w", "q", "x", "q!", "wq!", "x!"}:
            inserting = False
            continue
        if stripped == ".":
            inserting = False
            continue

        sub = re.match(r"(?:%s|1s|\d+s|\$s)/([^/]*)/([^/]*)/([g]?)$", ex)
        if sub:
            pattern, replacement, flags = sub.groups()
            substitutions.append((pattern, replacement, "g" in flags))
            continue

        if ex in {"i", "insert", "1i", "$i", "a", "append"}:
            inserting = True
            continue
        if ex.startswith("i") and len(ex) > 1 and ex not in {"insert"}:
            inserted.append(ex[1:])
            inserting = True
            continue

        if inserting or not stripped.startswith(":"):
            if line != "" or inserting:
                inserted.append(line)

    # A blank line is part of an insert block, but a trailing blank caused by
    # splitting on ESC/newline is not content.
    while inserted and inserted[-1] == "":
        inserted.pop()
    return inserted, substitutions


def parse_commands(commands: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, bool]]]:
    inserted: list[str] = []
    appended: list[str] = []
    substitutions: list[tuple[str, str, bool]] = []

    for raw in commands:
        cmd = clean(raw).strip()
        cmd = cmd[1:] if cmd.startswith(":") else cmd
        if cmd in {"wq", "w", "q", "x", "$"} or cmd.startswith("set "):
            continue

        match = re.match(r"(?:normal|norm)\s+([io])(.*)$", cmd)
        if match:
            mode, text = match.groups()
            text = text.strip()
            (inserted if mode == "i" else appended).append(text)
            continue

        match = re.match(r"(?:call\s+)?append\(\s*\d+\s*,\s*[\"'](.*)[\"']\s*\)", cmd)
        if match:
            inserted.append(match.group(1))
            continue

        match = re.match(r"(?:%s|1s|\d+s)/([^/]*)/([^/]*)/([g]?)$", cmd)
        if match:
            pattern, replacement, flags = match.groups()
            substitutions.append((pattern, replacement, "g" in flags))
            continue

        # Unhandled benign ex commands are ignored.  The benchmark's evaluator,
        # not this stub, decides whether the resulting file is correct.
    return inserted, appended, substitutions


def apply_substitutions(lines: list[str], subs: list[tuple[str, str, bool]]) -> list[str]:
    for pattern, replacement, _global in subs:
        if pattern == "^":
            lines = [replacement + line for line in lines] or [replacement]
        elif pattern == "$":
            lines = [line + replacement for line in lines] or [replacement]
        else:
            lines = [line.replace(pattern, replacement) for line in lines]
    return lines


def main() -> int:
    target, command_strings = parse_args(sys.argv[1:])
    if not target:
        return 0

    stdin_inserted, stdin_subs = parse_stdin(sys.stdin.read())
    cmd_inserted, cmd_appended, cmd_subs = parse_commands(command_strings)

    existing: list[str] = []
    if os.path.isfile(target):
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            existing = handle.read().splitlines()

    lines = apply_substitutions(existing, stdin_subs + cmd_subs)

    if stdin_inserted:
        # Interactive insert in these tasks replaces the touched (usually empty)
        # file, then a later `vi` invocation appends to it.
        lines = stdin_inserted
    else:
        before = cmd_inserted
        after = cmd_appended
        if before and not lines:
            lines = before + after
        elif before:
            # `normal i` inserts before the first line; `normal o` adds after it.
            lines = before + lines + after
        elif after:
            lines = lines + after

    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        if lines:
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
