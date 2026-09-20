"""Fresh capped pytest workers, aggregate memory limits, deadlines and durable results.

The supervisor imports neither pytest nor torch. macOS uses physical footprint
(including compressed memory), Linux uses RSS + swap. The sampled limit covers
the process group and discovered descendants. macOS also gets a kernel limit
per process through taskpolicy. Unsupported accounting fails closed.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
import ctypes
import errno
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

MIB, GIB = 1024**2, 1024**3
RESERVED_MACHINE_BYTES = 8 * GIB
DEFAULT_WORKER_MEMORY_BYTES = 8 * GIB
HERE = Path(__file__).resolve().parent
LOCK = Path.home() / ".cache" / "basicmodel" / "tests.lock"


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def suite_lock(path=LOCK):
    """One heavy suite per user, including suites in other worktrees."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A bounded test suite is already running ({path})") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


@contextmanager
def termination_signals():
    def interrupt(signum, frame):
        raise _Interrupted(signum)
    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


class _DarwinUsage(ctypes.Structure):
    # sys/resource.h: struct rusage_info_v0, RUSAGE_INFO_V0 = 0.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime")]


class _DarwinBSD(ctypes.Structure):
    # sys/proc_info.h: struct proc_bsdinfo (PROC_PIDTBSDINFO = 3).
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid",
        "rgid", "svuid", "svgid", "reserved")] + [
        ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)] + [
        (name, ctypes.c_uint32) for name in ("nfiles", "pgid", "pjobc", "tdev", "tpgid", "nice")] + [
        ("start_seconds", ctypes.c_uint64), ("start_microseconds", ctypes.c_uint64)]


class ProcessTree:
    """Account for descendants and retain birth identities to avoid PID reuse."""
    def __init__(self, pid):
        self.pid, self.known = pid, {}
        if sys.platform == "darwin":
            self.libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            self.libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self.libproc.proc_pid_rusage.restype = ctypes.c_int
            self.libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                                 ctypes.c_void_p, ctypes.c_int]
            self.libproc.proc_pidinfo.restype = ctypes.c_int
            self.libproc.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                                  ctypes.c_void_p, ctypes.c_int]
            self.libproc.proc_listpids.restype = ctypes.c_int
        elif not sys.platform.startswith("linux"):
            raise RuntimeError("Bounded tests require macOS or Linux memory accounting")
        self.memory(os.getpid())

    def darwin_process(self, pid):
        for attempt in range(6):
            info = _DarwinBSD()
            size = self.libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
            if size == ctypes.sizeof(info):
                return info
            error = ctypes.get_errno()
            if error in (errno.ESRCH, errno.ENOENT):
                return None
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            except PermissionError:
                pass
            if attempt == 5:
                raise OSError(error, f"Cannot read process identity for {pid}")
            time.sleep(.01)

    def darwin_pids(self, kind, identity):
        size = 256 * ctypes.sizeof(ctypes.c_int)
        while True:
            buffer = (ctypes.c_int * (size // ctypes.sizeof(ctypes.c_int)))()
            used = self.libproc.proc_listpids(kind, identity, buffer, size)
            if used < 0:
                raise OSError("Cannot enumerate worker processes")
            if used < size:
                return {pid for pid in buffer[:used // ctypes.sizeof(ctypes.c_int)] if pid}
            size *= 2

    def memory(self, pid):
        """Return (bytes, birth identity), or None for a vanished process."""
        if sys.platform == "darwin":
            usage = _DarwinUsage()
            for attempt in range(6):
                if self.libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
                    return usage.phys_footprint, usage.proc_start_abstime
                error = ctypes.get_errno()
                if error in (errno.ESRCH, errno.ENOENT):
                    return None
                info = self.darwin_process(pid)
                if info is None or info.status == 5 or info.flags & 4:
                    return None
                # Short setuid helpers (notably /bin/ps) are unreadable while
                # alive. Allow their exit race to settle, never ignore a live
                # unaccounted process for more than 50 ms.
                if attempt == 5:
                    raise OSError(error, f"{os.strerror(error)} (pid {pid})")
                time.sleep(.01)
        try:
            fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                      for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line}
            if fields["State"].startswith("Z"):
                return None
            size = sum(int(fields[key].split()[0]) * 1024 for key in ("VmRSS", "VmSwap"))
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return size, int(stat[19])  # /proc stat field 22, after comm + pid.
        except (FileNotFoundError, ProcessLookupError):
            return None

    def members(self):
        if sys.platform == "darwin":
            # Native group/child enumeration avoids both a setuid ps helper and
            # requests for unrelated processes' identities. PROC_PGRP_ONLY=2;
            # PROC_PPID_ONLY=6 also finds children that start their own session.
            candidates = self.darwin_pids(2, self.pid)
            for pid, birth in self.known.items():
                current = self.memory(pid)
                if current is not None and current[1] == birth:
                    candidates.add(pid)
            pending, visited = list(candidates), set()
            while pending:
                pid = pending.pop()
                if pid in visited:
                    continue
                visited.add(pid)
                children = self.darwin_pids(6, pid)
                candidates.update(children)
                pending.extend(children - visited)
            processes = []
            for pid in candidates:
                info = self.darwin_process(pid)
                if info is not None and info.status != 5 and not info.flags & 4:
                    processes.append((info.pid, info.ppid, info.pgid))
        else:
            rows = subprocess.check_output(
                ["ps", "-axo", "pid=,ppid=,pgid=,stat="], text=True, timeout=2)
            processes = [tuple(map(int, fields[:3])) for line in rows.splitlines()
                         if (fields := line.split()) and not fields[3].startswith("Z")]
        live_pids = {pid for pid, _, _ in processes}
        selected = {self.pid}
        for pid, birth in self.known.items():
            if pid not in live_pids:
                continue
            current = self.memory(pid)
            if current is not None and current[1] == birth:
                selected.add(pid)
        previous = None
        while previous != selected:
            previous = selected.copy()
            for pid, ppid, pgid in processes:
                if pgid == self.pid or ppid in selected:
                    selected.add(pid)
        return selected & live_pids

    def sample(self):
        total, live = 0, {}
        for pid in self.members():
            value = self.memory(pid)
            if value is not None:
                total += value[0]
                live[pid] = value[1]
        self.known = live
        return total

    def terminate(self, proc, grace):
        try:
            self.sample()  # Remember descendants before the leader is removed.
        except Exception:
            pass  # The process group remains available when the monitor fails.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.pid, sig)
            except (ProcessLookupError, PermissionError):
                # macOS can return EPERM when the group has only exited
                # members. Still signal known children and verify below.
                pass
            for pid, birth in self.known.items():
                try:
                    current = self.memory(pid)
                    if current is not None and current[1] == birth:
                        os.kill(pid, sig)
                except OSError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(grace)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        for _ in range(10):
            if not self.members():
                return
            time.sleep(.05)
        raise RuntimeError("Worker descendants remained alive after SIGKILL")


def bounded_command(command, memory_bytes):
    """Apply a kernel cap without putting a test worker in the background tier."""
    if sys.platform == "darwin":
        taskpolicy = shutil.which("taskpolicy")
        if taskpolicy is None:
            raise RuntimeError("taskpolicy is required for bounded macOS tests")
        return [taskpolicy, "-m", str(math.ceil(memory_bytes / MIB)), "-P", "kill",
                "nice", "-n", "10", *command]
    return ["nice", "-n", "10", *command]


class GuardedProcess:
    """One independently capped test worker that a suite can poll alongside peers."""

    def __init__(self, command, *, cwd, env, log_path, memory_bytes, timeout,
                 terminate_grace=1, recycle_file=None):
        for name, value in (("memory_bytes", memory_bytes), ("timeout", timeout),
                            ("terminate_grace", terminate_grace)):
            positive(value, name)
        self.command = list(command)
        self.cwd, self.env, self.log_path = cwd, env, Path(log_path)
        self.memory_bytes, self.timeout = memory_bytes, timeout
        self.terminate_grace, self.recycle_file = terminate_grace, recycle_file
        self.result = dict(command=self.command, pid=None, exit_code=125, reason="launch_error",
                           peak_memory_bytes=0, elapsed_seconds=0, log=str(self.log_path))
        self.proc = self.tree = self.log = None
        self.started = None
        self.finished = False
        self.current_memory_bytes = 0

    def start(self):
        """Launch the capped process group without blocking the caller."""
        if self.started is not None:
            raise RuntimeError("bounded worker was started twice")
        self.started = time.monotonic()
        try:
            self.log = self.log_path.open("w")
            self.tree = ProcessTree(0)
            launch_command = bounded_command(self.command, self.memory_bytes)
            self.proc = subprocess.Popen(launch_command, cwd=self.cwd, env=self.env,
                                         stdout=self.log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self.tree.pid = self.result["pid"] = self.proc.pid
        except Exception as exc:
            self.result.update(exit_code=125, reason="launch_error", error=str(exc))
            self.finish()
        return self

    def finish(self):
        """Kill surviving descendants, close diagnostics, and write one receipt."""
        if self.finished:
            return self.result
        self.finished = True
        if self.proc is not None:
            try:
                self.tree.terminate(self.proc, self.terminate_grace)
            except Exception as exc:
                self.result.update(exit_code=125, reason="cleanup_error", error=str(exc))
        if self.log is not None:
            self.log.close()
        self.result["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.log_path.with_suffix(".process.json"), self.result)
        return self.result

    def stop(self, *, exit_code, reason, error=None):
        """End an in-flight worker for a suite-level failure or interruption."""
        if not self.finished:
            self.result.update(exit_code=exit_code, reason=reason)
            if error is not None:
                self.result["error"] = error
        return self.finish()

    def poll(self):
        """Advance accounting once; return a receipt only after terminal cleanup."""
        if self.finished:
            return self.result
        try:
            used = self.tree.sample()
            self.current_memory_bytes = used
            self.result["peak_memory_bytes"] = max(used, self.result["peak_memory_bytes"])
            if used > self.memory_bytes:
                return self.stop(exit_code=137, reason="memory")
            elapsed = time.monotonic() - self.started
            # Large workers can retain work until 80% of their reservation;
            # small workers also keep a fixed allocator headroom so one more
            # test cannot jump straight into the kernel cap.
            recycle_memory = max(self.memory_bytes * .5,
                                 min(self.memory_bytes * .8,
                                     self.memory_bytes - 64 * MIB))
            if self.recycle_file is not None and (used >= recycle_memory
                                                  or elapsed >= self.timeout * .8):
                # Finish the current test before releasing a high-water worker.
                # The kernel/taskpolicy limit remains the hard backstop.
                if not Path(self.recycle_file).exists():
                    write_json(self.recycle_file, dict(
                        reason="memory_boundary" if used >= recycle_memory else "time_boundary",
                        measured_bytes=used, elapsed_seconds=elapsed))
            if elapsed >= self.timeout:
                return self.stop(exit_code=124, reason="timeout")
            exit_code = self.proc.poll()
            if exit_code is not None:
                return self.stop(exit_code=exit_code if exit_code >= 0 else 128 - exit_code,
                                 reason="exit" if exit_code >= 0 else "signal")
        except Exception as exc:
            return self.stop(exit_code=125, reason="monitor_error", error=str(exc))
        return None


def run_guarded(command, *, cwd, env, log_path, memory_bytes, timeout,
                poll_seconds=.25, terminate_grace=1, recycle_file=None):
    """Run one capped process group to completion; retain the standalone API."""
    positive(poll_seconds, "poll_seconds")
    guarded = GuardedProcess(
        command, cwd=cwd, env=env, log_path=log_path, memory_bytes=memory_bytes,
        timeout=timeout, terminate_grace=terminate_grace, recycle_file=recycle_file)
    with termination_signals():
        try:
            guarded.start()
            while guarded.poll() is None:
                remaining = max(0, guarded.timeout - (time.monotonic() - guarded.started))
                time.sleep(min(poll_seconds, remaining))
        except _Interrupted as exc:
            guarded.stop(exit_code=128 + exc.signum, reason="interrupted")
        except KeyboardInterrupt:
            guarded.stop(exit_code=130, reason="interrupted")
        except Exception as exc:
            guarded.stop(exit_code=125, reason="monitor_error", error=str(exc))
        finally:
            guarded.finish()
    return guarded.result


def make_batches(nodes, batch_size=256, max_files=16, devices=None):
    positive(batch_size, "batch_size")
    positive(max_files, "max_files")
    batches, batch, files = [], [], set()
    for node in nodes:
        filename = node.split("::", 1)[0]
        if batch and (len(batch) >= batch_size
                      or (filename not in files and len(files) >= max_files)
                      or (devices is not None and devices[node] != devices[batch[0]])):
            batches.append(batch)
            batch, files = [], set()
        batch.append(node)
        files.add(filename)
    if batch:
        batches.append(batch)
    return batches


def source_snapshot(root):
    paths = set()
    for directory, suffixes in (("bin", {".py"}), ("test", {".py"}),
                                ("data", {".xml", ".xsd", ".grammar", ".json"})):
        paths.update(p for p in (root / directory).rglob("*") if p.suffix in suffixes and p.is_file())
    paths.update(root / name for name in ("pytest.ini", "Makefile", "requirements.txt", "README.md")
                 if (root / name).is_file())
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def documentation_snapshot(root):
    """Record mutable prose without letting a receipt block a documentation edit."""
    root = Path(root)
    paths = {p for p in (root / "doc").rglob("*") if p.suffix == ".md" and p.is_file()}
    if (root / "todo.md").is_file():
        paths.add(root / "todo.md")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def requires_suite_lock(selectors, keyword, marker):
    """Only a complete receipt claims the user-wide full-suite slot."""
    return not selectors and keyword is None and marker is None


def worker_environment(root):
    env = os.environ.copy()
    # The receipt worker always runs one pytest process; direct `make testp`
    # owns xdist iteration and may retain TEST_JOBS in its own environment.
    env.pop("TEST_JOBS", None)
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS")})
    # Significant training is opt-in and uses an accelerator. Preserve a
    # caller's explicit device for targeted compatibility checks.
    env.setdefault("BASICMODEL_DEVICE", "gpu" if env.get("RUN_SLOW") == "1" else "cpu")
    env["PYTHONUNBUFFERED"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONPATH"] = os.pathsep.join([str(root / "bin"), str(root / "test"), env.get("PYTHONPATH", "")])
    return env


def render_report(path, result):
    rows = []
    for worker in result["workers"]:
        for report in worker.get("reports", []):
            rows.append("<tr>" + "".join(f"<td><pre>{html.escape(str(report.get(k, '')))}</pre></td>"
                        for k in ("nodeid", "phase", "outcome", "duration", "message", "stdout")) + "</tr>")
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>BasicModel tests</title>"
        "<style>body{font:14px system-ui;margin:2em}td,th{border:1px solid #ccc;padding:.5em}"
        "table{border-collapse:collapse}pre{white-space:pre-wrap;max-width:50em}</style></head><body>"
        f"<h1>Tests: {html.escape(result['reason'])}</h1><p>Exit {result['exit_code']}; "
        f"{len(result['completed'])}/{len(result['selected'])} selected cases completed.</p>"
        "<p>Collection, worker logs, process limits and complete coverage are recorded in result.json.</p>"
        "<table><tr><th>Test</th><th>Phase</th><th>Outcome</th><th>Seconds</th><th>Details</th><th>Output</th></tr>"
        + "".join(rows) + "</table></body></html>")


def run_suite(*, root, selectors, run_dir, memory_bytes, timeout=1800, suite_timeout=10800,
              batch_size=256, max_files=16, workers=1, worker_memory_bytes=None,
              keyword=None, marker=None, lock_path=LOCK):
    """Collect once, then run fresh workers under one aggregate memory reservation.

    ``memory_bytes`` is the total suite reservation. Each worker has an
    independently enforced ceiling (8 GiB by default), while the supervisor
    samples all active process trees and fails before their aggregate crosses
    the reservation. A single accelerator worker is admitted at a time because
    MPS/CUDA use shared device memory.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    for name, value in (("memory_bytes", memory_bytes), ("timeout", timeout),
                        ("suite_timeout", suite_timeout), ("batch_size", batch_size),
                        ("max_files", max_files)):
        positive(value, name)
    if worker_memory_bytes is None:
        worker_memory_bytes = min(memory_bytes, DEFAULT_WORKER_MEMORY_BYTES)
    positive(worker_memory_bytes, "per_worker_memory_bytes")
    if worker_memory_bytes > memory_bytes:
        raise ValueError("per-worker memory cap cannot exceed the aggregate reservation")
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    env = worker_environment(root)
    split_devices = env.get("RUN_SLOW") == "1" and not os.environ.get("BASICMODEL_DEVICE")
    devices, active = {}, {}
    run_dir.mkdir(parents=True, exist_ok=False)  # Never reuse a stale success receipt.
    started, frozen = time.monotonic(), None
    result = dict(exit_code=125, reason="incomplete", selected=[], completed=[], workers=[],
                  limits=dict(memory_bytes=memory_bytes, aggregate_memory_bytes=memory_bytes,
                              per_worker_memory_bytes=worker_memory_bytes, workers=workers,
                              worker_seconds=timeout, suite_seconds=suite_timeout,
                              batch_size=batch_size, max_files=max_files), root=str(root),
                  run_dir=str(run_dir), elapsed_seconds=0,
                  requested_device=env["BASICMODEL_DEVICE"],
                  device_policy=({"ordinary": "cpu", "slow": env["BASICMODEL_DEVICE"]}
                                 if split_devices else {"all": env["BASICMODEL_DEVICE"]}))

    def save():
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(run_dir / "result.json", result)
        render_report(run_dir / "report.html", result)

    def publish_active():
        entries = [dict(name=handle["name"], progress_file=str(handle["response_file"]),
                        device=handle["device"])
                   for handle in active.values()]
        if entries:
            result["active_workers"] = entries
            if len(entries) == 1:
                result["active_worker"] = entries[0]
            else:
                result.pop("active_worker", None)
        else:
            result.pop("active_worker", None)
            result.pop("active_workers", None)
        save()

    def launch_worker(name, request, cap):
        remaining = suite_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Suite deadline reached")
        if source_snapshot(root) != frozen:
            raise RuntimeError("Tested source changed during this run; validation is incomplete")
        request_file, response_file = run_dir / f"{name}.request.json", run_dir / f"{name}.json"
        recycle_file = None if request.get("collect") else run_dir / f"{name}.recycle.json"
        request["recycle_file"] = str(recycle_file) if recycle_file is not None else None
        worker_env = env.copy()
        if split_devices:
            worker_env["BASICMODEL_DEVICE"] = (
                "cpu" if request.get("collect") else devices[request["selectors"][0]])
        write_json(request_file, request)
        print(f"[{name}] {len(request['selectors'])} selectors; limit {min(timeout, remaining):.0f}s, "
              f"{cap/GIB:.1f} GiB", flush=True)
        guarded = GuardedProcess(
            [sys.executable, str(HERE / "pytest_worker.py"), str(request_file), str(response_file)],
            cwd=root, env=worker_env, log_path=run_dir / f"{name}.log",
            memory_bytes=cap, timeout=min(timeout, remaining), recycle_file=recycle_file)
        guarded.start()
        return dict(name=name, request=request, response_file=response_file,
                    guarded=guarded, device=worker_env["BASICMODEL_DEVICE"])

    def settle(handle):
        receipt = handle["guarded"].result
        if handle["response_file"].exists():
            receipt["pytest"] = json.loads(handle["response_file"].read_text())
            receipt["device"] = receipt["pytest"].get("device")
        print(f"[{handle['name']}] {receipt['reason']} {receipt['exit_code']}; "
              f"{receipt['elapsed_seconds']:.1f}s; peak {receipt['peak_memory_bytes']/GIB:.2f} GiB", flush=True)
        return receipt

    def wait_for(handle):
        while handle["guarded"].poll() is None:
            result["peak_aggregate_memory_bytes"] = max(
                result.get("peak_aggregate_memory_bytes", 0),
                handle["guarded"].current_memory_bytes)
            remaining = max(0, suite_timeout - (time.monotonic() - started))
            if remaining <= 0:
                raise TimeoutError("Suite deadline reached")
            time.sleep(min(.05, remaining))
        return settle(handle)

    def account(nodes, receipt):
        data = receipt.pop("pytest", {})
        receipt.update(selected=data.get("selected", []), completed=data.get("completed", []),
                       reports=data.get("reports", []), recycled=data.get("recycled", False))
        result["workers"].append(receipt)
        result["completed"].extend(receipt["completed"])
        test_failure = any(report["outcome"] in ("failed", "xpassed")
                           for report in receipt["reports"])
        # pytest uses status 1 for an ordinary assertion/setup/teardown failure.
        # It is diagnostic information, not a failed process boundary: keep
        # dispatching the selected cases so one receipt reports every failure.
        # Timeouts, memory kills, monitor failures, collection/protocol errors
        # and other non-pytest exits still stop the pool immediately.
        if receipt["exit_code"] != 0 and not (receipt["exit_code"] == 1 and test_failure):
            result.update(exit_code=receipt["exit_code"], reason=receipt["reason"])
            return False
        if test_failure:
            result.update(exit_code=1, reason="test_failure")
        completed = receipt["completed"]
        if receipt["recycled"]:
            if not completed or completed != nodes[:len(completed)]:
                result.update(exit_code=125, reason="invalid_recycle_coverage")
                return False
            if len(completed) < len(nodes):
                batches.appendleft(nodes[len(completed):])
        elif Counter(completed) != Counter(nodes):
            result.update(exit_code=125, reason="incomplete_coverage")
            return False
        if Counter(receipt["selected"]) != Counter(nodes):
            result.update(exit_code=125, reason="incomplete_coverage")
            return False
        return True

    def uses_accelerator(device):
        return device != "cpu"

    def abort_active(reason):
        for handle in list(active.values()):
            handle["guarded"].stop(exit_code=125, reason=reason)
            settle(handle)
        active.clear()
        publish_active()

    save()
    batches = deque()
    try:
        lock_context = (suite_lock(lock_path) if requires_suite_lock(selectors, keyword, marker)
                        else nullcontext())
        with lock_context, termination_signals():
            frozen = source_snapshot(root)
            documentation = documentation_snapshot(root)
            write_json(run_dir / "source-manifest.json", dict(
                validated_source=frozen, recorded_documentation=documentation))
            collect_handle = launch_worker(
                "collect", dict(selectors=selectors or ["test"], collect=True,
                                keyword=keyword, marker=marker), worker_memory_bytes)
            active["collect"] = collect_handle
            publish_active()
            collect = wait_for(collect_handle)
            active.pop("collect", None)
            publish_active()
            result["collection"] = collect
            result["selected"] = collect.get("pytest", {}).get("selected", [])
            if collect["exit_code"] != 0:
                result.update(exit_code=collect["exit_code"], reason="collection_" + collect["reason"])
            elif not result["selected"]:
                result.update(exit_code=5, reason="empty_selection")
            else:
                result.update(exit_code=125, reason="running")
                slow = set(collect.get("pytest", {}).get("slow_selected", []))
                devices = {node: env["BASICMODEL_DEVICE"] if not split_devices or node in slow
                           else "cpu" for node in result["selected"]}
                batches = deque(make_batches(result["selected"], batch_size, max_files, devices))
                index = 0
                while batches or active:
                    remaining = suite_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Suite deadline reached")
                    while len(active) < workers and batches:
                        accelerator_live = any(uses_accelerator(handle["device"])
                                               for handle in active.values())
                        candidate = next(
                            (position for position, nodes in enumerate(batches)
                             if not (uses_accelerator(devices[nodes[0]]) and accelerator_live)),
                            None)
                        if candidate is None:
                            break
                        nodes = batches[candidate]
                        del batches[candidate]
                        name = f"worker-{index:03d}"
                        index += 1
                        active[name] = launch_worker(name, dict(selectors=nodes, collect=False),
                                                     worker_memory_bytes)
                        publish_active()
                    finished = []
                    for name, handle in list(active.items()):
                        if handle["guarded"].poll() is not None:
                            finished.append((name, handle))
                    active_memory = sum(handle["guarded"].current_memory_bytes
                                        for handle in active.values()
                                        if not handle["guarded"].finished)
                    result["peak_aggregate_memory_bytes"] = max(
                        result.get("peak_aggregate_memory_bytes", 0), active_memory)
                    if active_memory > memory_bytes:
                        name, handle = max(
                            ((name, handle) for name, handle in active.items()
                             if not handle["guarded"].finished),
                            key=lambda item: item[1]["guarded"].current_memory_bytes)
                        handle["guarded"].stop(exit_code=137, reason="aggregate_memory")
                        finished.append((name, handle))
                    if not finished:
                        time.sleep(.05)
                        continue
                    stop_dispatch = False
                    for name, handle in finished:
                        active.pop(name, None)
                        receipt = settle(handle)
                        if not account(handle["request"]["selectors"], receipt):
                            abort_active("aborted")
                            stop_dispatch = True
                            break
                        publish_active()
                    if stop_dispatch:
                        break
                if result["reason"] in ("running", "test_failure"):
                    if Counter(result["completed"]) == Counter(result["selected"]):
                        if result["reason"] == "running":
                            result.update(exit_code=0, reason="passed")
                    else:
                        result.update(exit_code=125, reason="incomplete_coverage")
            if source_snapshot(root) != frozen:
                result.update(exit_code=125, reason="source_changed")
    except _Interrupted as exc:
        result.update(exit_code=128 + exc.signum, reason="interrupted")
        abort_active("interrupted")
    except TimeoutError as exc:
        result.update(exit_code=124, reason="suite_timeout", error=str(exc))
        abort_active("suite_timeout")
    except Exception as exc:
        result.update(exit_code=125, reason="runner_error", error=str(exc))
        abort_active("aborted")
    finally:
        if active:
            abort_active("aborted")
        save()
    return result


def physical_memory():
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2))
    if sys.platform.startswith("linux"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    raise RuntimeError("Bounded tests require macOS or Linux")


def default_test_memory_bytes():
    """Reserve eight GiB for the interactive machine, not half of all RAM."""
    available = physical_memory() - RESERVED_MACHINE_BYTES
    if available <= 0:
        raise ValueError("Bounded tests require more than 8 GiB of physical RAM")
    return available


def default_worker_count():
    """Use all but four CPU slots so SSH and interactive work stay responsive."""
    return max(1, (os.cpu_count() or 1) - 4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selectors", nargs="*", help="pytest files or node IDs; default: complete test/ suite")
    parser.add_argument("-k", dest="keyword")
    parser.add_argument("-m", dest="marker")
    parser.add_argument("--memory-gib", type=float,
                        help="aggregate test memory; default leaves 8 GiB for the machine")
    parser.add_argument("--workers", type=int, default=default_worker_count(),
                        help="concurrent one-thread workers; default reserves CPU headroom")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per worker, including collection")
    parser.add_argument("--suite-timeout", type=float, default=10800, help="overall seconds")
    parser.add_argument("--batch-size", type=int, default=256, help="maximum test cases per fresh worker")
    parser.add_argument("--max-files", type=int, default=16, help="maximum test files per fresh worker")
    parser.add_argument("--run-dir", type=Path, help="new directory for logs, report and durable result.json")
    args = parser.parse_args(argv)
    root = HERE.parent
    run_dir = args.run_dir or root / "output" / "tests" / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    try:
        memory_bytes = (default_test_memory_bytes() if args.memory_gib is None
                        else int(args.memory_gib * GIB))
        positive(memory_bytes, "memory-gib")
        if memory_bytes > default_test_memory_bytes():
            raise ValueError("Test memory limit must leave at least 8 GiB for the machine")
        result = run_suite(root=root, selectors=args.selectors, run_dir=run_dir,
                           memory_bytes=memory_bytes, timeout=args.timeout,
                           suite_timeout=args.suite_timeout, batch_size=args.batch_size,
                           max_files=args.max_files, workers=args.workers,
                           keyword=args.keyword, marker=args.marker)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Result: {result['reason']}, exit {result['exit_code']}; "
          f"{len(result['completed'])}/{len(result['selected'])} cases completed.\n{run_dir / 'result.json'}", flush=True)
    return result["exit_code"], run_dir / "report.html"
