"""Fresh sequential pytest workers, memory limits, deadlines and durable results.

The supervisor imports neither pytest nor torch. macOS uses physical footprint
(including compressed memory), Linux uses RSS + swap. The sampled limit covers
the process group and discovered descendants. macOS also gets a kernel limit
per process through taskpolicy. Unsupported accounting fails closed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
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


def run_guarded(command, *, cwd, env, log_path, memory_bytes, timeout,
                poll_seconds=.25, terminate_grace=1, recycle_file=None):
    """Run one bounded process group; clean up before reporting completion."""
    for name, value in (("memory_bytes", memory_bytes), ("timeout", timeout),
                        ("poll_seconds", poll_seconds), ("terminate_grace", terminate_grace)):
        positive(value, name)
    started = time.monotonic()
    result = dict(command=list(command), pid=None, exit_code=125, reason="launch_error",
                  peak_memory_bytes=0, elapsed_seconds=0, log=str(log_path))
    proc = tree = None
    with termination_signals(), Path(log_path).open("w") as log:
        try:
            tree = ProcessTree(0)
            bounded_command = list(command)
            if sys.platform == "darwin":
                taskpolicy = shutil.which("taskpolicy")
                if taskpolicy is None:
                    raise RuntimeError("taskpolicy is required for bounded macOS tests")
                bounded_command = [taskpolicy, "-b", "-m", str(math.ceil(memory_bytes / MIB)),
                                   "-P", "kill", *bounded_command]
            else:
                bounded_command = ["nice", "-n", "10", *bounded_command]
            proc = subprocess.Popen(bounded_command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            tree.pid = result["pid"] = proc.pid
            while True:
                used = tree.sample()
                result["peak_memory_bytes"] = max(used, result["peak_memory_bytes"])
                if used > memory_bytes:
                    result.update(exit_code=137, reason="memory")
                    break
                elapsed = time.monotonic() - started
                if recycle_file is not None and (used >= memory_bytes / 2 or elapsed >= timeout / 2):
                    # Ask pytest to finish the current case, then release its
                    # process. The hard limit remains in force meanwhile.
                    if not Path(recycle_file).exists():
                        write_json(recycle_file, dict(
                            reason="memory_boundary" if used >= memory_bytes / 2 else "time_boundary",
                            measured_bytes=used, elapsed_seconds=elapsed))
                if elapsed >= timeout:
                    result.update(exit_code=124, reason="timeout")
                    break
                exit_code = proc.poll()
                if exit_code is not None:
                    result.update(exit_code=exit_code if exit_code >= 0 else 128 - exit_code,
                                  reason="exit" if exit_code >= 0 else "signal")
                    break
                time.sleep(min(poll_seconds, max(0, timeout - (time.monotonic() - started))))
        except _Interrupted as exc:
            result.update(exit_code=128 + exc.signum, reason="interrupted")
        except KeyboardInterrupt:
            result.update(exit_code=130, reason="interrupted")
        except Exception as exc:
            result.update(exit_code=125, reason="monitor_error" if proc else "launch_error", error=str(exc))
        finally:
            if proc is not None:
                try:
                    tree.terminate(proc, terminate_grace)
                except Exception as exc:
                    result.update(exit_code=125, reason="cleanup_error", error=str(exc))
            result["elapsed_seconds"] = time.monotonic() - started
    write_json(Path(log_path).with_suffix(".process.json"), result)
    return result


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
                                ("data", {".xml", ".xsd", ".grammar"}), ("doc", {".md"})):
        paths.update(p for p in (root / directory).rglob("*") if p.suffix in suffixes and p.is_file())
    paths.update(root / name for name in ("pytest.ini", "Makefile", "requirements.txt", "README.md", "todo.md")
                 if (root / name).is_file())
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def worker_environment(root):
    env = os.environ.copy()
    if env.get("TEST_JOBS", "1") not in ("", "1"):
        raise ValueError("TEST_JOBS must be 1; use selected test files for fast bounded iterations")
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
              batch_size=256, max_files=16, keyword=None, marker=None, lock_path=LOCK):
    """Collect once; run every selected node exactly once in fresh serial workers."""
    for name, value in (("memory_bytes", memory_bytes), ("timeout", timeout),
                        ("suite_timeout", suite_timeout), ("batch_size", batch_size), ("max_files", max_files)):
        positive(value, name)
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    env = worker_environment(root)
    split_devices = env.get("RUN_SLOW") == "1" and not os.environ.get("BASICMODEL_DEVICE")
    devices = {}
    run_dir.mkdir(parents=True, exist_ok=False)  # Never reuse a stale success receipt.
    started = time.monotonic()
    result = dict(exit_code=125, reason="incomplete", selected=[], completed=[], workers=[],
                  limits=dict(memory_bytes=memory_bytes, worker_seconds=timeout, suite_seconds=suite_timeout,
                              batch_size=batch_size, max_files=max_files), root=str(root),
                  run_dir=str(run_dir), elapsed_seconds=0,
                  requested_device=env["BASICMODEL_DEVICE"],
                  device_policy=({"ordinary": "cpu", "slow": env["BASICMODEL_DEVICE"]}
                                 if split_devices else {"all": env["BASICMODEL_DEVICE"]}))

    def save():
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(run_dir / "result.json", result)
        render_report(run_dir / "report.html", result)

    def worker(name, request):
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
        result["active_worker"] = dict(name=name, progress_file=str(response_file),
                                      device=worker_env["BASICMODEL_DEVICE"])
        write_json(request_file, request)
        save()
        print(f"[{name}] {len(request['selectors'])} selectors; limit {min(timeout, remaining):.0f}s, "
              f"{memory_bytes/GIB:.1f} GiB", flush=True)
        receipt = run_guarded(
            [sys.executable, str(HERE / "pytest_worker.py"), str(request_file), str(response_file)],
            cwd=root, env=worker_env, log_path=run_dir / f"{name}.log",
            memory_bytes=memory_bytes, timeout=min(timeout, remaining), recycle_file=recycle_file)
        if response_file.exists():
            receipt["pytest"] = json.loads(response_file.read_text())
            receipt["device"] = receipt["pytest"].get("device")
        result.pop("active_worker", None)
        print(f"[{name}] {receipt['reason']} {receipt['exit_code']}; "
              f"{receipt['elapsed_seconds']:.1f}s; peak {receipt['peak_memory_bytes']/GIB:.2f} GiB", flush=True)
        return receipt

    save()
    try:
        with suite_lock(lock_path), termination_signals():
            frozen = source_snapshot(root)
            write_json(run_dir / "source-manifest.json", frozen)
            collect = worker("collect", dict(selectors=selectors or ["test"], collect=True,
                                            keyword=keyword, marker=marker))
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
                batches = make_batches(result["selected"], batch_size, max_files, devices)
                index = 0
                while batches:
                    nodes = batches.pop(0)
                    receipt = worker(f"worker-{index:03d}", dict(selectors=nodes, collect=False))
                    index += 1
                    data = receipt.pop("pytest", {})
                    receipt.update(selected=data.get("selected", []), completed=data.get("completed", []),
                                   reports=data.get("reports", []), recycled=data.get("recycled", False))
                    result["workers"].append(receipt)
                    result["completed"].extend(receipt["completed"])
                    if receipt["exit_code"] != 0:
                        result.update(exit_code=receipt["exit_code"], reason=receipt["reason"])
                        break
                    if any(r["outcome"] in ("failed", "xpassed") for r in receipt["reports"]):
                        result.update(exit_code=1, reason="test_failure")
                        break
                    completed = receipt["completed"]
                    if receipt["recycled"]:
                        if not completed or completed != nodes[:len(completed)]:
                            result.update(exit_code=125, reason="invalid_recycle_coverage")
                            break
                        if len(completed) < len(nodes):
                            batches.insert(0, nodes[len(completed):])
                    elif Counter(completed) != Counter(nodes):
                        result.update(exit_code=125, reason="incomplete_coverage")
                        break
                    if Counter(receipt["selected"]) != Counter(nodes):
                        result.update(exit_code=125, reason="incomplete_coverage")
                        break
                    save()
                if result["reason"] == "running":
                    if Counter(result["completed"]) == Counter(result["selected"]):
                        result.update(exit_code=0, reason="passed")
                    else:
                        result.update(exit_code=125, reason="incomplete_coverage")
            if source_snapshot(root) != frozen:
                result.update(exit_code=125, reason="source_changed")
    except _Interrupted as exc:
        result.update(exit_code=128 + exc.signum, reason="interrupted")
    except TimeoutError as exc:
        result.update(exit_code=124, reason="suite_timeout", error=str(exc))
    except Exception as exc:
        result.update(exit_code=125, reason="runner_error", error=str(exc))
    finally:
        save()
    return result


def physical_memory():
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2))
    if sys.platform.startswith("linux"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    raise RuntimeError("Bounded tests require macOS or Linux")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selectors", nargs="*", help="pytest files or node IDs; default: complete test/ suite")
    parser.add_argument("-k", dest="keyword")
    parser.add_argument("-m", dest="marker")
    parser.add_argument("--memory-gib", type=float, default=min(8, physical_memory() / GIB / 3))
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per worker, including collection")
    parser.add_argument("--suite-timeout", type=float, default=10800, help="overall seconds")
    parser.add_argument("--batch-size", type=int, default=256, help="maximum test cases per fresh worker")
    parser.add_argument("--max-files", type=int, default=16, help="maximum test files per fresh worker")
    parser.add_argument("--run-dir", type=Path, help="new directory for logs, report and durable result.json")
    args = parser.parse_args(argv)
    root = HERE.parent
    run_dir = args.run_dir or root / "output" / "tests" / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    try:
        positive(args.memory_gib, "memory-gib")
        if args.memory_gib * GIB > physical_memory() / 2:
            raise ValueError("Test memory limit must leave at least half of physical RAM for the machine")
        result = run_suite(root=root, selectors=args.selectors, run_dir=run_dir,
                           memory_bytes=int(args.memory_gib * GIB), timeout=args.timeout,
                           suite_timeout=args.suite_timeout, batch_size=args.batch_size,
                           max_files=args.max_files, keyword=args.keyword, marker=args.marker)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Result: {result['reason']}, exit {result['exit_code']}; "
          f"{len(result['completed'])}/{len(result['selected'])} cases completed.\n{run_dir / 'result.json'}", flush=True)
    return result["exit_code"], run_dir / "report.html"
