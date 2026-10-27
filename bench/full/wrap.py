"""Run a command and record what it cost: `wrap.py OUT.json -- CMD...`.

The command runs as this process's child. Once it exits, os.wait4 gives its rusage, and
the rusage covers every descendant it waited for (its worker pools, its decoder). That is
the stage's CPU time (user and system), its largest single process (maxrss), its block
I/O (on Linux, inblock and oublock are the task's read_bytes and write_bytes over 512),
its page faults and context switches. OUT gets these with the command's start, end and
exit code, and this process exits with the command's code. A signal sent to this process
(SIGTERM, SIGINT) is passed on to the command.

Run it with `python -I`, so it imports nothing from the environment it passes on.
"""

import json
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    out, sep, *cmd = sys.argv[1:]
    assert sep == "--" and cmd, __doc__
    t0, w0 = time.time(), time.perf_counter()
    proc = subprocess.Popen(cmd)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: proc.send_signal(s))
    while True:
        try:
            _, status, ru = os.wait4(proc.pid, 0)
            break
        except InterruptedError:
            continue
    wall = time.perf_counter() - w0
    code = os.waitstatus_to_exitcode(status)
    proc.returncode = code  # (so Popen does not wait for it again)
    rec = {
        "cmd": cmd, "pid": proc.pid, "rc": code, "t0": t0, "t1": time.time(), "wall": round(wall, 3),
        "user": round(ru.ru_utime, 3), "sys": round(ru.ru_stime, 3), "cpu": round(ru.ru_utime + ru.ru_stime, 3),
        "maxrss_mb": round(ru.ru_maxrss / 1024, 1), "read_bytes": ru.ru_inblock * 512,
        "write_bytes": ru.ru_oublock * 512, "minflt": ru.ru_minflt, "majflt": ru.ru_majflt,
        "nvcsw": ru.ru_nvcsw, "nivcsw": ru.ru_nivcsw,
    }
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
