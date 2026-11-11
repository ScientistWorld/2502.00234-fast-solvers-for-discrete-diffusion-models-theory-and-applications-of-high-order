"""Run registered solvers in a separate process against an exact oracle.

The parent process owns the instance.  It answers each intensity request,
charges ceil(m / n) function evaluations for a request of m rows (n being the
number of samples), refuses requests once the NFE budget is spent, validates
every request and the returned samples, and enforces a time limit.  Any
violation, exception or malformed output is a failed run.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import time
from multiprocessing import Pipe
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKER = HERE / "worker.py"
SOLVER_ROOTS = ("method", "baseline")
MAX_ROWS_FACTOR = 4


class SolverFailure(RuntimeError):
    pass


def _spawn():
    parent, child = Pipe(duplex=True)
    proc = subprocess.Popen(
        [sys.executable, "-I", str(WORKER), str(child.fileno()), str(REPO_ROOT)],
        pass_fds=[child.fileno()],
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr.fileno(),
        stderr=sys.stderr.fileno(),
    )
    child.close()
    return parent, proc


def _close(conn, proc) -> None:
    try:
        conn.close()
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _send(conn, header) -> None:
    conn.send_bytes(json.dumps(header).encode())


def _recv(conn, deadline: float):
    """Receive a JSON header.  Messages from solvers are never unpickled."""
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not conn.poll(remaining):
        raise SolverFailure("time limit exceeded")
    try:
        msg = json.loads(conn.recv_bytes(maxlength=65536).decode())
    except (EOFError, OSError) as exc:
        raise SolverFailure("solver process exited without a result") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise SolverFailure("malformed message from solver") from exc
    if not isinstance(msg, list) or not msg:
        raise SolverFailure("malformed message from solver")
    return msg


def _recv_bytes(conn, deadline: float, size: int) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not conn.poll(remaining):
        raise SolverFailure("time limit exceeded")
    try:
        buf = conn.recv_bytes(maxlength=size + 1)
    except (EOFError, OSError) as exc:
        raise SolverFailure("malformed message from solver") from exc
    if len(buf) != size:
        raise SolverFailure("message size mismatch")
    return buf


def _has_solver_registry(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return "SOLVERS" in path.read_text(errors="ignore")
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "SOLVERS" for t in targets):
            return True
    return False


def solver_files(roots: tuple[str, ...] = SOLVER_ROOTS) -> list[str]:
    files = []
    for root in roots:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if _has_solver_registry(path):
                files.append(str(path.relative_to(REPO_ROOT)))
    return files


def list_solvers(file: str, timeout: float = 120.0) -> list[str]:
    conn, proc = _spawn()
    try:
        _send(conn, {"mode": "list", "file": file})
        reply = _recv(conn, time.monotonic() + timeout)
        if reply[0] != "names" or len(reply) < 2 or not isinstance(reply[1], list):
            raise SolverFailure(str(reply[-1])[-1500:])
        return [str(name) for name in reply[1]]
    finally:
        _close(conn, proc)


def run_solver(
    file: str,
    name: str,
    oracle,
    initial_states: np.ndarray,
    nfe_budget: int,
    seed: int,
    timeout: float,
) -> tuple[np.ndarray, dict]:
    inst = oracle.inst
    n, d = initial_states.shape
    S, T = inst.num_states, inst.horizon
    spec = {
        "mode": "run",
        "file": file,
        "name": name,
        "num_samples": n,
        "num_sites": d,
        "num_states": S,
        "horizon": T,
        "nfe_budget": int(nfe_budget),
        "seed": int(seed),
    }
    start = time.monotonic()
    deadline = start + timeout
    conn, proc = _spawn()
    nfe = 0
    calls = 0
    ok = False
    try:
        _send(conn, spec)
        conn.send_bytes(np.ascontiguousarray(initial_states, dtype=np.int64).tobytes())
        while True:
            msg = _recv(conn, deadline)
            kind = msg[0]
            if kind == "rates":
                m = int(msg[1])
                if m < 0 or m > MAX_ROWS_FACTOR * n:
                    raise SolverFailure(f"request of {m} rows is outside [0, {MAX_ROWS_FACTOR}n]")
                states = np.frombuffer(_recv_bytes(conn, deadline, m * d * 8), dtype=np.int64)
                states = states.reshape(m, d)
                if msg[2] is None:
                    times = np.frombuffer(_recv_bytes(conn, deadline, m * 8), dtype=np.float64)
                else:
                    times = np.float64(msg[2])
                if m and (states.min() < 0 or states.max() >= S):
                    raise SolverFailure("requested states out of range")
                if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times > T):
                    raise SolverFailure("requested reverse time outside [0, horizon]")
                nfe += math.ceil(m / n) if m else 0
                calls += 1
                if nfe > nfe_budget:
                    _send(conn, ["refused", "NFE budget exceeded"])
                    raise SolverFailure(f"NFE budget {nfe_budget} exceeded")
                rates = oracle.rates(states, times)
                _send(conn, ["ok", nfe])
                conn.send_bytes(np.ascontiguousarray(rates, dtype=np.float64).tobytes())
            elif kind == "done":
                if int(msg[1]) != n or int(msg[2]) != d:
                    raise SolverFailure("solver returned the wrong number of samples")
                out = np.frombuffer(_recv_bytes(conn, deadline, n * d * 8), dtype=np.int64)
                out = out.reshape(n, d).copy()
                if out.min() < 0 or out.max() >= S:
                    raise SolverFailure("returned states out of range")
                ok = True
                return out, {
                    "nfe_used": nfe,
                    "oracle_calls": calls,
                    "seconds": round(time.monotonic() - start, 3),
                }
            elif kind == "error":
                raise SolverFailure(f"solver raised: {str(msg[1])[-1500:]}")
            else:
                raise SolverFailure("unknown message from solver")
    finally:
        if not ok and proc.poll() is None:
            proc.kill()
        _close(conn, proc)
