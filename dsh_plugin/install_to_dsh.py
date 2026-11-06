#!/usr/bin/env python3
"""Register the audio-perception MCP server into a local DSH profile (idempotent).

Appends a `@deepseek-ai/dsh-mcp-client` entry to
  ~/.dsh/profiles/<profile>/cordis.patch.yml
with a timestamped backup, then prints restart instructions.

This writes OUTSIDE the workspace (~/.dsh); run it yourself, or approve the
sandbox escalation when prompted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import yaml

ENTRY_ID = "dsh-audio-perception"
PROJECT = Path(__file__).resolve().parent.parent


class TolerantLoader(yaml.SafeLoader):
    pass


def _unknown(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


TolerantLoader.add_multi_constructor("!", _unknown)


def entry_yaml() -> str:
    return (
        f"- id: {ENTRY_ID}\n"
        f'  name: "@deepseek-ai/dsh-mcp-client"\n'
        f"  config:\n"
        f"    serverName: audio\n"
        f"    transport: stdio\n"
        f"    command: {PROJECT}/.venv/bin/python\n"
        f'    args: ["{PROJECT}/audio_mcp_server.py"]\n'
        f"    cwd: {PROJECT}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="desktop", help="DSH profile name (default: desktop)")
    ap.add_argument("--dsh-home", default=str(Path.home() / ".dsh"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patch = Path(args.dsh_home) / "profiles" / args.profile / "cordis.patch.yml"
    if not patch.exists():
        print(f"error: profile patch not found: {patch}", file=sys.stderr)
        return 1
    text = patch.read_text(encoding="utf-8")
    if ENTRY_ID in text:
        print(f"already installed in {patch}; nothing to do.")
        return 0

    new_text = text.rstrip("\n") + "\n\n" + entry_yaml()
    # validate tolerant YAML
    try:
        data = yaml.load(new_text, Loader=TolerantLoader)
        assert isinstance(data, list)
    except Exception as e:
        print(f"error: refusing to write, YAML did not parse: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(new_text[-600:])
        print("\n[dry-run] not written.")
        return 0

    backup = patch.with_suffix(patch.suffix + f".bak-{dt.datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(patch, backup)
    patch.write_text(new_text, encoding="utf-8")
    print(f"installed '{ENTRY_ID}' into {patch}\nbackup: {backup}")
    print("\nRestart the DSH app (or the target profile) to load the MCP server.")
    print("Then the agent gets tools: mcp__audio__analyze_audio / transcribe_audio / detect_emotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
