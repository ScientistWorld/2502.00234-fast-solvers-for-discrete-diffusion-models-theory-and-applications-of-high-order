"""Solver process for the test harness.

Started by eval/test/harness.py as `python -I worker.py <fd> <repo_root>`.
It imports one registered solver, hands it a `Problem` whose only channel to
the instance is `rates()`, and returns the final samples.  Instance data never
enters this process: every intensity request is answered by the parent, which
also counts the function evaluations.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path


def _send(conn, header) -> None:
    import json

    conn.send_bytes(json.dumps(header).encode())


def _recv(conn):
    import json

    return json.loads(conn.recv_bytes().decode())


class Problem:
    """What a solver sees.  See briefing/problem.md for the contract."""

    def __init__(self, conn, spec: dict, initial_states):
        import numpy as np

        self._conn = conn
        self.num_samples = int(spec["num_samples"])
        self.num_sites = int(spec["num_sites"])
        self.num_states = int(spec["num_states"])
        self.horizon = float(spec["horizon"])
        self.nfe_budget = int(spec["nfe_budget"])
        self.seed = int(spec["seed"])
        self.initial_states = initial_states
        self.rng = np.random.default_rng(self.seed)
        self.nfe_used = 0

    def rates(self, states, s):
        import numpy as np

        x = np.asarray(states)
        if x.ndim != 2 or x.shape[1] != self.num_sites:
            raise ValueError(f"states must have shape (m, {self.num_sites})")
        if not np.issubdtype(x.dtype, np.integer):
            if x.size and (not np.all(np.isfinite(x)) or not np.all(x == np.round(x))):
                raise ValueError("states must be integer valued")
        x = np.ascontiguousarray(x, dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= self.num_states):
            raise ValueError("states out of range")
        m = x.shape[0]
        times = np.asarray(s, dtype=np.float64)
        if times.ndim not in (0, 1) or (times.ndim == 1 and times.shape != (m,)):
            raise ValueError("s must be a scalar or have shape (m,)")
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times > self.horizon):
            raise ValueError("reverse time must lie in [0, horizon]")
        per_row = times.ndim == 1
        _send(self._conn, ["rates", m, None if per_row else float(times)])
        self._conn.send_bytes(x.tobytes())
        if per_row:
            self._conn.send_bytes(np.ascontiguousarray(times).tobytes())
        reply = _recv(self._conn)
        if reply[0] != "ok":
            raise RuntimeError(f"intensity request refused: {reply[1]}")
        self.nfe_used = int(reply[1])
        buf = self._conn.recv_bytes()
        return np.frombuffer(buf, dtype=np.float64).reshape(
            m, self.num_sites, self.num_states
        ).copy()


def _load(root: Path, file: str):
    path = (root / file).resolve()
    rel = path.relative_to(root).with_suffix("")
    return importlib.import_module(".".join(rel.parts))


def main() -> None:
    fd = int(sys.argv[1])
    root = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(root))
    from multiprocessing.connection import Connection

    conn = Connection(fd)
    request = _recv(conn)
    try:
        import numpy as np

        module = _load(root, request["file"])
        solvers = getattr(module, "SOLVERS", None)
        if not isinstance(solvers, dict):
            raise TypeError("SOLVERS must be a dict mapping names to callables")
        if request["mode"] == "list":
            _send(conn, ["names", [str(k) for k in solvers]])
            return
        fn = solvers[request["name"]]
        n, d = int(request["num_samples"]), int(request["num_sites"])
        x0 = np.frombuffer(conn.recv_bytes(), dtype=np.int64).reshape(n, d).copy()
        out = np.asarray(fn(Problem(conn, request, x0)))
        if out.shape != (n, d):
            raise ValueError(f"solver returned shape {out.shape}, expected {(n, d)}")
        if not np.issubdtype(out.dtype, np.integer):
            if not np.all(np.isfinite(out)) or not np.all(out == np.round(out)):
                raise ValueError("solver returned non-integer states")
        out = np.ascontiguousarray(out, dtype=np.int64)
        _send(conn, ["done", n, d])
        conn.send_bytes(out.tobytes())
    except BaseException:
        try:
            _send(conn, ["error", traceback.format_exc()[-4000:]])
        except Exception:
            pass


if __name__ == "__main__":
    main()
