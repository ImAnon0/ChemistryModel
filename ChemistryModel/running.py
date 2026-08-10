import json
import os
import signal
import subprocess
import sys
import time


# ============================================================
# Knowing what is still running
# ============================================================
#
# Batches are separate processes, so closing the control panel
# leaves them going. That is the behaviour worth having, but it
# means the panel has to work out on startup which of the jobs it
# remembers are still alive - otherwise it would cheerfully start
# a second copy of each, and two processes writing the same index
# would destroy it.
#
# Each batch writes a small lock file into its output folder when
# it starts and removes it when it finishes. The lock holds the
# process id and when it began. On startup the panel reads any
# lock it finds and checks whether that process is genuinely
# still there.
#
# A hard kill leaves the lock behind, so a lock alone is not
# trusted: the process has to still exist, and the index has to
# have been touched recently enough to be plausible.


LOCK_NAME = ".running"

# If the index has not been written to in this long, a lock is
# treated as left over from something that died.

STALE_AFTER = 3600.0


def lock_path(folder):
    return os.path.join(folder, LOCK_NAME)


def write_lock(folder, command=None):
    os.makedirs(folder, exist_ok=True)

    with open(lock_path(folder), "w") as handle:
        json.dump({
            "pid": os.getpid(),
            "started": time.time(),
            "command": command or [],
        }, handle)


def remove_lock(folder):
    try:
        os.remove(lock_path(folder))
    except OSError:
        pass


def read_lock(folder):
    path = lock_path(folder)

    if not os.path.exists(path):
        return None

    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def is_alive(pid):
    if not pid:
        return False

    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except ImportError:
        pass

    if os.name == "nt":
        try:
            output = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout

            return str(pid) in output
        except Exception:
            return False

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, but it does exist.
        return True
    except (OSError, ValueError):
        return False

    return True


def index_touched(folder):
    path = os.path.join(folder, "index.json")

    if not os.path.exists(path):
        return 0.0

    return os.path.getmtime(path)


def state_of(folder):
    # Returns one of "running", "stale" or "gone", plus the lock.

    lock = read_lock(folder)

    if lock is None:
        return "gone", None

    if not is_alive(lock.get("pid")):
        return "stale", lock

    touched = index_touched(folder)

    if touched and time.time() - touched > STALE_AFTER:
        # The process exists but has not written anything for a
        # long time. Could be a very slow run, could be something
        # wedged; either way it is worth flagging rather than
        # assuming.
        return "stale", lock

    return "running", lock


def stop(pid, timeout=3.0):
    # Terminate first so the batch can finish writing whatever it
    # is partway through, and only force it if that is ignored.

    if not is_alive(pid):
        return True

    try:
        import psutil

        process = psutil.Process(int(pid))

        process.terminate()

        try:
            process.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            process.kill()

        return True
    except ImportError:
        pass
    except Exception:
        return False

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(int(pid))],
            capture_output=True,
        )

        deadline = time.time() + timeout

        while time.time() < deadline and is_alive(pid):
            time.sleep(0.2)

        if is_alive(pid):
            subprocess.run(
                ["taskkill", "/F", "/PID", str(int(pid))],
                capture_output=True,
            )

        return True

    try:
        os.kill(int(pid), signal.SIGTERM)

        deadline = time.time() + timeout

        while time.time() < deadline and is_alive(pid):
            time.sleep(0.2)

        if is_alive(pid):
            os.kill(int(pid), signal.SIGKILL)
    except (OSError, ValueError):
        return False

    return True
