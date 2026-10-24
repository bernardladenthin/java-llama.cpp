<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# RPC COMM_INIT: red/green regression record

Verified on Linux x86-64, 2026-10-24, at llama.cpp b11827 (`970b51ce7`), using the real
C API shared by JNI and FFM. All scenarios use CPU, local TCP sockets and no model download.

## The defect

The old `0015` patch stopped the main listener/client but did not track a COMM_INIT peer.
Rank 0 listened on `0.0.0.0` even when the main server bound `127.0.0.1`. With no peer it
blocked in accept; with an incomplete peer it blocked reading transport capabilities.
Rank 1 could likewise block waiting for capabilities. Repeated C-API stop calls also opened
wake-up connections to the unserved main listener and could themselves block behind its backlog.

## Same tests, before and after the fix

`llama/src/test/cpp/test_rpc_comm.cpp` was added to `jllama_test` before changing product code.
The red run used the original `0015` from parent commit `672a9238`. The green run uses its
replacement from this change; the six scenarios are the same.

| scenario | old patch | fixed patch |
|---|---|---|
| local server rejects a peer connection to a second loopback address | fails: wildcard listener | passes |
| network bind remains an explicit opt-in | passes | passes |
| stop without a peer, free both ports, restart on both | fails: stop/serve deadline | passes |
| stop during rank-0 capability handshake | fails: stop/serve deadline | passes |
| stop during rank-1 capability handshake | fails: stop/serve deadline | passes |
| successful peer setup still works, stops and permits restart | passes | passes |

Red: **4 failed, 2 passed** (15.4 seconds). Green: **6 passed** (0.57 seconds).
The focused suite including existing RPC/C-API cases passed **36/36** (1.33 seconds).
The full C++ suite also passed **692/692** tests. All six new scenarios additionally passed
ten consecutive repetitions each (60 executions).
Times are observations from this host, not timing guarantees.

Each scenario re-executes a fresh GoogleTest subprocess, has bounded I/O and a watchdog,
and bounds stop/serve completion to five seconds. A broken shutdown fails the child without
leaving live threads in the parent. The test never supplies a recovery peer to make stop pass.
It observes listener readiness through existing debug logs; active bind probes were removed
because they can race the listener and occupy its port. Every readiness timeout is a failure.
The two address-isolation cases explicitly skip if the host cannot bind `127.0.0.2`; the four
shutdown/normal-setup cases still run. There were **no skips** in the recorded Linux run.

## Fix and maintenance

- The peer listener inherits the configured main-server bind address.
- Main/peer accept and outgoing peer connect support cancellation through short socket polls
  (`poll` on POSIX, `WSAPoll` on Windows); accepted/connected sockets return to blocking I/O.
- Peer sockets are weakly registered under the stop mutex before capability I/O. Stop shuts
  them down together with the main client, including peers that race with stop.
- Normal COMM_FREE/disconnect releases peers immediately; the registry does not own them.
- The rank-1 retry loop observes stop. Main-listener wake-up connections are removed.
- Shared transport initialization remains alive so stopping one server does not invalidate
  other RPC clients in the process.

The change is carried in `llama/patches/0015-rpc-embeddable-client-and-stoppable-server.patch`.
All twelve repository patches were applied together to a pristine upstream pin, through the
normal CMake patch applier. No downloaded build tree is the sole copy of the fix.

## Running the regression cases

After configuring the native build as described in the project's build instructions:

```bash
cmake --build llama/build --target jllama_test -j 4
ctest --test-dir llama/build -R '^RpcComm\.' --output-on-failure --timeout 30
```

The CMake source list includes these cases in the existing C++ CI jobs. Windows/macOS
runtime results are still required from CI; this local result proves Linux behaviour.
This is model-free coverage of RPC setup/shutdown, not a distributed tensor-inference benchmark.
