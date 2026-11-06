#!/usr/bin/env python3
"""Self-test the audio MCP server over stdio: initialize, tools/list, analyze one file."""
import subprocess, json, sys, pathlib, select
PYBIN = str(pathlib.Path(__file__).resolve().parent.parent / ".venv/bin/python")
SRV = str(pathlib.Path(__file__).resolve().parent.parent / "audio_mcp_server.py")
target = sys.argv[1] if len(sys.argv) > 1 else (print("usage: selftest.py <audio_file>") or sys.exit(2))
p = subprocess.Popen([PYBIN, "-u", SRV], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
def send(o): p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
def recv(t=600):
    r, _, _ = select.select([p.stdout], [], [], t)
    return p.stdout.readline()
send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}); recv()
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
print("tools:", [t["name"] for t in json.loads(recv())["result"]["tools"]])
for scheme in ("5class", "multilabel"):
    send({"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "analyze_audio", "arguments": {"path": target, "emotion_scheme": scheme}}})
    r = json.loads(recv())["result"]
    if r.get("isError"):
        print(scheme, "ERROR:", r["content"][0]["text"])
    else:
        d = json.loads(r["content"][0]["text"])
        print(scheme, "->", d.get("transcript"), "|", d.get("emotion") or d.get("labels"))
p.terminate()
