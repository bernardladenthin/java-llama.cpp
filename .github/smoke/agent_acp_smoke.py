# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Drive the agent release jar as an editor would, over the Agent Client Protocol.

Starts the given command (the agent with --acp), speaks newline-delimited JSON-RPC 2.0 on its
stdin/stdout the way JetBrains IDEs and Zed do, and checks what an editor would show:

  1. initialize and session/new answer, with the approval modes and the slash commands;
  2. a plain prompt streams an answer containing "4" and ends with stopReason end_turn;
  3. a prompt that needs read_file produces a tool_call of kind "read" and brings the marker
     from the session's working directory back.

Only the standard library, so it runs on a stock runner. Everything the agent sent is written
to the log file given as the second argument, its stderr next to it as <name>.err.log.

Usage: agent_acp_smoke.py <workspace> <log> -- <command...>
"""

import json
import os
import subprocess
import sys
import threading
import time

TIMEOUT = float(os.environ.get("AGENT_SMOKE_TIMEOUT", "600"))


def fail(message):
    print(f"::error::{message}", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        fail("usage: agent_acp_smoke.py <workspace> <log> -- <command...>")
    workspace, log_path, command = sys.argv[1], sys.argv[2], sys.argv[4:]
    err_path = (log_path[:-4] if log_path.endswith(".log") else log_path) + ".err.log"
    marker = f"ACP_SMOKE_{int(time.time())}"
    with open(os.path.join(workspace, "marker.txt"), "w", encoding="utf-8") as f:
        f.write(marker + "\n")

    with open(err_path, "w", encoding="utf-8") as err_log:
        agent = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err_log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
    received = []
    lock = threading.Condition()

    def read():
        with open(log_path, "w", encoding="utf-8") as log:
            for line in agent.stdout:
                log.write(line)
                log.flush()
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    # stdout is the protocol: anything else on it would break a real editor
                    message = {"__not_json__": line}
                with lock:
                    received.append(message)
                    lock.notify_all()

    threading.Thread(target=read, daemon=True).start()
    ids = iter(range(1, 1000))

    def send(message):
        agent.stdin.write(json.dumps(message) + "\n")
        agent.stdin.flush()

    def request(method, params):
        request_id = next(ids)
        send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return request_id

    def response(request_id):
        deadline = time.time() + TIMEOUT
        with lock:
            while time.time() < deadline:
                for message in received:
                    if "__not_json__" in message:
                        fail(f"the agent wrote something other than JSON-RPC to stdout: {message['__not_json__']!r}")
                    if message.get("id") == request_id and "method" not in message:
                        if "error" in message:
                            fail(f"request {request_id} failed: {message['error']}")
                        return message["result"]
                    if message.get("method") == "session/request_permission":
                        # nothing this smoke asks for is gated; a question means the gate moved
                        fail(f"unexpected permission request: {message}")
                if agent.poll() is not None:
                    fail(f"the agent exited with {agent.returncode} before answering request {request_id}")
                lock.wait(1)
        fail(f"no answer to request {request_id} within {TIMEOUT:.0f}s")

    def updates(kind, since):
        with lock:
            return [
                m["params"]["update"]
                for m in received[since:]
                if m.get("method") == "session/update" and m["params"]["update"].get("sessionUpdate") == kind
            ]

    try:
        init = response(request("initialize", {"protocolVersion": 1, "clientCapabilities": {}}))
        if init.get("protocolVersion") != 1:
            fail(f"unexpected protocol version: {init}")
        session = response(request("session/new", {"cwd": workspace, "mcpServers": []}))
        session_id = session["sessionId"]
        modes = [m["id"] for m in session["modes"]["availableModes"]]
        if modes != ["manual", "auto"]:
            fail(f"unexpected session modes: {modes}")
        print(f"OK: initialize + session/new (modes {modes})")

        def prompt(text):
            since = len(received)
            result = response(
                request("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]})
            )
            answer = "".join(u["content"].get("text", "") for u in updates("agent_message_chunk", since))
            return result, answer, since

        result, answer, _ = prompt("What is 2 + 2? Answer with one short sentence.")
        if result.get("stopReason") != "end_turn":
            fail(f"plain prompt ended with {result}")
        if "4" not in answer:
            fail(f"the answer does not contain 4: {answer!r}")
        print(f"OK: plain answer over ACP ({answer.strip()[:80]!r})")

        commands = [c["name"] for c in updates("available_commands_update", 0)[-1]["availableCommands"]]
        if "status" not in commands:
            fail(f"slash commands were not announced: {commands}")
        print(f"OK: {len(commands)} slash commands announced")

        result, answer, since = prompt("Read the file marker.txt with the read_file tool and tell me its exact content.")
        calls = updates("tool_call", since)
        if not any(c.get("kind") == "read" and c.get("title", "").startswith("read_file") for c in calls):
            fail(f"no read_file tool call: {calls}")
        seen = answer + json.dumps(updates("tool_call_update", since))
        if marker not in seen:
            fail(f"the marker never reached the editor: {answer!r}")
        print("OK: tool round over ACP (read_file)")
    finally:
        agent.stdin.close()
        try:
            agent.wait(timeout=30)
        except subprocess.TimeoutExpired:
            agent.kill()
            fail("the agent did not exit after the editor closed stdin")
    if agent.returncode != 0:
        fail(f"the agent exited with {agent.returncode} after the editor closed stdin")
    print("OK: exits cleanly when the editor hangs up")


if __name__ == "__main__":
    main()
