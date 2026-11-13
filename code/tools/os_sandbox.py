"""A tiny, Docker-free execution sandbox for LifelongAgentBench OS tasks.

Each task gets a fresh user/mount/PID namespace and a chroot-like filesystem.
Host `/usr` is bind-mounted read-only; `/etc`, `/home`, `/var`, `/tmp`, etc. are
per-task copies.  Ground-truth/evaluation or agent scripts run against a
throwaway root, without Docker or root privileges on the host.

Because unprivileged user namespaces normally map only uid/gid 0, a bundled
`fakeroot` daemon keeps fake ownership metadata.  This lets `chown`, `stat`,
`useradd`, etc. behave like they do in the paper's Ubuntu container.

Linux x86-64 only.  Requires util-linux `unshare` and `mount`.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
VENDOR_DIR = TOOLS_DIR / "vendor"
# Rootless WSL user namespaces may bind-mount directories owned by the user,
# but not the system-owned /usr tree. A WSL-specific staging helper can create
# a user-owned snapshot and point this variable at it. Other Linux hosts retain
# the original /usr behavior by default.
_default_staged_usr = Path.home() / ".cache" / "lab_8b_kit" / "usr"
HOST_USR = Path(
    os.environ.get(
        "OS_SANDBOX_HOST_USR",
        str(_default_staged_usr if _default_staged_usr.is_dir() else Path("/usr")),
    )
).resolve()
VI_STUB = TOOLS_DIR / "vi_stub.py"
FAKED_BIN = VENDOR_DIR / "faked-sysv"
FAKEROOT_LIB = VENDOR_DIR / "libfakeroot-sysv.so"
FORCE_EPERM_LIB = VENDOR_DIR / "libforce_chown_eperm.so"

ETC_FILES = [
    "passwd",
    "group",
    "shells",
    "login.defs",
    "nsswitch.conf",
    "hosts",
    "host.conf",
    "resolv.conf",
    "os-release",
    "profile",
    "bash.bashrc",
    "environment",
    "subuid",
    "subgid",
    "adduser.conf",
    "deluser.conf",
    "securetty",
]
ETC_DIRS = ["pam.d", "security", "skel", "default", "alternatives", "ssl"]


def _ignore_unreadable(path: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if not os.access(os.path.join(path, name), os.R_OK)
    }


def prepare_root(rootfs: Path) -> None:
    for name in [
        "usr",
        "bin",
        "sbin",
        "etc",
        "proc",
        "dev",
        "tmp",
        "run",
        "root",
        "home",
        "var",
        "opt",
        "srv",
        "mnt",
        "media",
    ]:
        (rootfs / name).mkdir(parents=True, exist_ok=True)

    etc = rootfs / "etc"
    for name in ETC_FILES:
        src = Path("/etc") / name
        if src.is_file() and os.access(src, os.R_OK):
            shutil.copy2(src, etc / name, follow_symlinks=True)
    for name in ETC_DIRS:
        src = Path("/etc") / name
        if src.is_dir():
            shutil.copytree(
                src,
                etc / name,
                dirs_exist_ok=True,
                symlinks=True,
                ignore=_ignore_unreadable,
                ignore_dangling_symlinks=True,
            )

    # Mimic the Ubuntu image's minimal skel files.
    (etc / "skel").mkdir(exist_ok=True)
    (etc / "skel" / ".bashrc").write_text(
        "# ~/.bashrc: executed by bash(1) for non-login shells.\n", encoding="utf-8"
    )
    (etc / "skel" / ".profile").write_text(
        "# ~/.profile: executed by the command interpreter for login shells.\n",
        encoding="utf-8",
    )
    (etc / "skel" / ".bash_logout").write_text("", encoding="utf-8")

    # The sandbox host often has a login user at uid 1000 (e.g. "user").
    # The paper's Ubuntu image does not; remove it so benchmark tasks can
    # create users with those names/uids.
    host_user = pwd.getpwuid(os.getuid()).pw_name

    def keep_system_account(line: str) -> bool:
        parts = line.split(":")
        if len(parts) < 3 or not parts[2].isdigit():
            return True
        uid_or_gid = int(parts[2])
        return parts[0] != host_user and not (1000 <= uid_or_gid < 65534)

    passwd_lines = [
        line
        for line in (etc / "passwd").read_text(encoding="utf-8").splitlines()
        if line.strip() and keep_system_account(line)
    ]
    sanitized_groups: list[str] = []
    for line in (etc / "group").read_text(encoding="utf-8").splitlines():
        if not line.strip() or not keep_system_account(line):
            continue
        parts = line.split(":")
        if len(parts) >= 4:
            members = [m for m in parts[-1].split(",") if m and m != host_user]
            parts[-1] = ",".join(members)
        sanitized_groups.append(":".join(parts))
    group_lines = sanitized_groups
    (etc / "passwd").write_text("\n".join(passwd_lines) + "\n", encoding="utf-8")
    (etc / "group").write_text("\n".join(group_lines) + "\n", encoding="utf-8")
    for name in ["subuid", "subgid"]:
        path = etc / name
        if path.exists():
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith(f"{host_user}:")
            ]
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (etc / "shadow").write_text(
        "\n".join(f"{line.split(':', 1)[0]}:*:20000:0:99999:7:::" for line in passwd_lines) + "\n",
        encoding="utf-8",
    )
    (etc / "gshadow").write_text(
        "\n".join(f"{line.split(':', 1)[0]}:*::" for line in group_lines) + "\n",
        encoding="utf-8",
    )
    os.chmod(etc / "shadow", 0o640)
    os.chmod(etc / "gshadow", 0o640)

    shells = set((etc / "shells").read_text(encoding="utf-8").splitlines())
    shells.update({"/bin/sh", "/bin/bash", "/bin/zsh", "/usr/bin/zsh", "/bin/dash"})
    (etc / "shells").write_text("\n".join(sorted(shells)) + "\n", encoding="utf-8")
    (etc / "hostname").write_text("osbench\n", encoding="utf-8")
    (etc / "hosts").write_text(
        "127.0.0.1 localhost osbench\n::1 localhost ip6-localhost ip6-loopback\n",
        encoding="utf-8",
    )
    (etc / "resolv.conf").write_text("nameserver 127.0.0.11\n", encoding="utf-8")

    # Permissive PAM: the namespace root is not the host root.
    pam_dir = etc / "pam.d"
    pam_dir.mkdir(exist_ok=True)
    permissive = "\n".join(
        f"{kind} sufficient pam_permit.so"
        for kind in ["auth", "account", "password", "session"]
    ) + "\n"
    for service in ["chsh", "chage", "passwd", "useradd", "groupadd"]:
        (pam_dir / service).write_text(permissive, encoding="utf-8")

    for name in ["root", "home", "var", "opt", "srv", "mnt", "media", "run"]:
        os.chmod(rootfs / name, 0o755)
    os.chmod(rootfs / "tmp", 0o1777)
    for subdir in ["log", "tmp", "cache", "lib", "spool", "mail", "opt", "backups"]:
        (rootfs / "var" / subdir).mkdir(parents=True, exist_ok=True)
    (rootfs / "var" / "tmp").chmod(0o1777)
    for dev, target in [
        ("fd", "/proc/self/fd"),
        ("stdin", "/proc/self/fd/0"),
        ("stdout", "/proc/self/fd/1"),
        ("stderr", "/proc/self/fd/2"),
    ]:
        link = rootfs / "dev" / dev
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    (rootfs / "dev" / "shm").mkdir(exist_ok=True)
    for dev in ["null", "zero", "random", "urandom"]:
        (rootfs / "dev" / dev).touch(exist_ok=True)


def _mount(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mount failed: {' '.join(args)}\n{result.stdout}")


def _setup_namespace(rootfs: Path) -> tuple[dict[str, str], int]:
    """Create mounts, chroot, start faked; return (script env, faked pid)."""
    usr = rootfs / "usr"
    # A writable tmpfs over /usr/local prevents any setup write from touching
    # the read-only, bind-mounted host /usr snapshot.
    if not (HOST_USR / "bin").is_dir() or not (HOST_USR / "lib").is_dir():
        raise RuntimeError(f"host /usr snapshot is incomplete: {HOST_USR}")
    _mount(["mount", "--bind", str(HOST_USR), str(usr)])
    _mount(["mount", "-t", "tmpfs", "tmpfs", str(usr / "local")])
    local_bin = usr / "local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)

    for src, name in [
        (VI_STUB, "vi"),
        (FAKED_BIN, "faked-sysv"),
    ]:
        shutil.copy2(src, local_bin / name)
        os.chmod(local_bin / name, 0o755)
    (local_bin / "vim").symlink_to("vi")
    (local_bin / "apt-get").write_text(
        "#!/bin/bash\n# Offline stand-in: no package installation is needed.\nexit 0\n",
        encoding="utf-8",
    )
    os.chmod(local_bin / "apt-get", 0o755)
    (local_bin / "apt").symlink_to("apt-get")
    (local_bin / "zsh").write_text("#!/bin/bash\nexec /bin/bash \"$@\"\n", encoding="utf-8")
    os.chmod(local_bin / "zsh", 0o755)
    # fakeroot does not emulate kernel setgid-directory group inheritance.
    # The published OS tasks rely on it, so this wrapper explicitly assigns a
    # newly touched file the fake group of its setgid parent directory.
    (local_bin / "touch").write_text(
        """#!/bin/bash
new_files=()
for arg in "$@"; do
    [[ "$arg" != -* && ! -e "$arg" ]] && new_files+=("$arg")
done
/host-bin/touch "$@"
rc=$?
if [[ $rc -eq 0 && ${#new_files[@]} -gt 0 ]]; then
    for path in "${new_files[@]}"; do
        [[ -e "$path" ]] || continue
        parent=$(dirname "$path")
        mode=$(stat -c '%a' "$parent" 2>/dev/null || echo 0)
        if [[ "$mode" =~ ^[2367][0-7]{3}$ ]]; then
            group=$(stat -c '%G' "$parent" 2>/dev/null || echo UNKNOWN)
            [[ "$group" != UNKNOWN ]] && chgrp "$group" "$path" 2>/dev/null || true
        fi
    done
fi
exit $rc
""",
        encoding="utf-8",
    )
    os.chmod(local_bin / "touch", 0o755)
    # Perl's taint mode (used by Debian's addgroup) rejects world-writable
    # PATH directories. tmpfs mounts default to 1777; make them 0755.
    for directory in [usr / "local", local_bin]:
        os.chmod(directory, 0o755)

    fakeroot_lib_dir = usr / "local" / "lib" / "fakeroot"
    fakeroot_lib_dir.mkdir(parents=True, exist_ok=True)
    for src in [FAKEROOT_LIB, FORCE_EPERM_LIB]:
        shutil.copy2(src, fakeroot_lib_dir / src.name)
    (usr / "local" / "sbin").mkdir(parents=True, exist_ok=True)

    # Expose host executables at /host-bin and /host-sbin, then overlay
    # /usr/bin and /usr/sbin with writable symlink farms.  The Ubuntu image in
    # the paper has a writable /usr/bin; benchmark scripts create helper files
    # there.  This keeps those writes inside the throwaway namespace.
    host_mounts = [
        (str(HOST_USR / "bin"), "host-bin"),
        (str(HOST_USR / "sbin"), "host-sbin"),
    ]
    for host_dir, mount_name in host_mounts:
        host_mount = rootfs / mount_name
        host_mount.mkdir(exist_ok=True)
        _mount(["mount", "--bind", host_dir, str(host_mount)])
        _mount(["mount", "-o", "remount,bind,ro", str(host_mount)])
    for host_dir, usr_subdir, mount_name in [
        (str(HOST_USR / "bin"), "bin", "host-bin"),
        (str(HOST_USR / "sbin"), "sbin", "host-sbin"),
    ]:
        overlay = usr / usr_subdir
        _mount(["mount", "-t", "tmpfs", "tmpfs", str(overlay)])
        for name in os.listdir(host_dir):
            source = os.path.join(host_dir, name)
            if not os.path.exists(source) or os.path.isdir(source):
                continue
            link = overlay / name
            if os.path.lexists(link):
                continue
            link.symlink_to(f"/{mount_name}/{name}")

    # /bin and /sbin are writable symlink farms too.
    for usr_subdir, dest in [("bin", rootfs / "bin"), ("sbin", rootfs / "sbin")]:
        for name in os.listdir(HOST_USR / usr_subdir):
            source = os.path.join(str(HOST_USR / usr_subdir), name)
            target = dest / name
            if os.path.lexists(target) or not os.path.exists(source):
                continue
            target.symlink_to(f"/usr/{usr_subdir}/{name}")
    for directory in [
        rootfs / "bin",
        rootfs / "sbin",
        usr / "bin",
        usr / "sbin",
        usr / "local" / "sbin",
    ]:
        if directory.exists():
            os.chmod(directory, 0o755)
    for zsh_path in [rootfs / "bin" / "zsh", usr / "bin" / "zsh"]:
        if zsh_path.exists() or zsh_path.is_symlink():
            zsh_path.unlink()
        zsh_path.symlink_to("/usr/local/bin/zsh")
    for link_name, link_target in [("lib", "usr/lib"), ("lib64", "usr/lib64")]:
        link = rootfs / link_name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(link_target)

    _mount(["mount", "-o", "remount,bind,ro", str(usr)])
    _mount(["mount", "-t", "proc", "proc", str(rootfs / "proc")])
    for dev in ["null", "zero", "random", "urandom"]:
        _mount(["mount", "--bind", f"/dev/{dev}", str(rootfs / "dev" / dev)])

    os.chroot(str(rootfs))
    os.chdir("/root")
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "SHELL": "/bin/bash",
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    faked = subprocess.run(
        ["/usr/local/bin/faked-sysv"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    lines = (faked.stdout or "").strip().splitlines()
    if faked.returncode != 0 or not lines or ":" not in lines[0]:
        raise RuntimeError(f"faked failed: rc={faked.returncode} out={faked.stdout!r} err={faked.stderr!r}")
    key, pid_text = lines[0].split(":", 1)
    faked_pid = int(pid_text)
    env.update(
        {
            "FAKEROOTKEY": key,
            "FAKED_MODE": "unknown-is-root",
            "LD_LIBRARY_PATH": "/usr/local/lib/fakeroot",
            # libfakeroot comes first; its RTLD_NEXT chown calls hit the shim,
            # which returns EPERM, so it stores fake ownership instead.
            "LD_PRELOAD": (
                "/usr/local/lib/fakeroot/libfakeroot-sysv.so "
                "/usr/local/lib/fakeroot/libforce_chown_eperm.so"
            ),
        }
    )
    return env, faked_pid


def _run_one(script: str, timeout: float, env: dict[str, str]) -> dict[str, Any]:
    """Run one script with a wall-clock timeout and no inherited stdout pipe.

    A temporary file avoids waiting for EOF from background daemons.  Those
    daemons deliberately remain alive across turns (some benchmark tasks check
    them later); the enclosing PID namespace kills them when the task ends.
    """
    started = time.perf_counter()
    output_path = Path(tempfile.mkstemp(prefix="cmd-", suffix=".log", dir="/tmp")[1])
    timed_out = False
    exit_code: int | None = None
    with output_path.open("w+b") as output_file:
        process = subprocess.Popen(
            ["/bin/bash", "-c", script],
            cwd="/root",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait(timeout=5)
            exit_code = None
        # Do not kill the process group here: several OS tasks intentionally
        # launch background work and check it in a later turn.  The enclosing
        # PID namespace cleans everything up when the sandbox closes.
        output_file.flush()
    output_bytes = output_path.read_bytes()[-8000:]
    output_path.unlink(missing_ok=True)
    return {
        "exit_code": exit_code,
        "output": output_bytes.decode("utf-8", "replace"),
        "timeout_flag": timed_out,
        "seconds": round(time.perf_counter() - started, 4),
    }


def inner_server(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rootfs = Path(spec["rootfs"])
    faked_pid: int | None = None
    try:
        env, faked_pid = _setup_namespace(rootfs)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        sys.exit(1)

    sys.stdout.reconfigure(line_buffering=True)
    print(json.dumps({"type": "ready"}), flush=True)
    command_timeout = float(spec["command_timeout"])
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"type": "error", "error": "invalid JSON request"}), flush=True)
            break
        if request.get("type") == "stop":
            break
        result = _run_one(request.get("script", ""), command_timeout, env)
        print(json.dumps({"type": "result", **result}), flush=True)
    if faked_pid:
        try:
            os.kill(faked_pid, signal.SIGTERM)
        except OSError:
            pass


class InteractiveSandbox:
    """A stateful sandbox; multiple scripts share one throwaway OS instance."""

    def __init__(self, *, command_timeout: float = 20.0):
        for path in [
            Path("/usr/bin/unshare"),
            Path("/usr/bin/mount"),
            VI_STUB,
            FAKED_BIN,
            FAKEROOT_LIB,
            FORCE_EPERM_LIB,
        ]:
            if not path.exists():
                raise RuntimeError(f"missing required file/tool: {path}")
        self.rootfs = Path(tempfile.mkdtemp(prefix="osbench-", dir=tempfile.gettempdir()))
        prepare_root(self.rootfs)
        self.spec_path = self.rootfs / "spec.json"
        self.spec_path.write_text(
            json.dumps(
                {
                    "rootfs": str(self.rootfs),
                    "command_timeout": command_timeout,
                }
            ),
            encoding="utf-8",
        )
        command = [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--kill-child",
            "--propagation",
            "private",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "--inner-server",
            str(self.spec_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.started = time.perf_counter()
        first = self._readline(15.0)
        if not first:
            raise RuntimeError("sandbox did not start; see outer_log")
        message = json.loads(first)
        if message.get("type") != "ready":
            raise RuntimeError(f"sandbox startup failed: {message}")
        self.closed = False

    def _readline(self, timeout: float) -> str | None:
        assert self.process.stdout is not None
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            return None
        return self.process.stdout.readline()

    def execute(self, script: str, *, timeout: float | None = None) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("sandbox is closed")
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"type": "exec", "script": script}) + "\n")
        self.process.stdin.flush()
        line = self._readline(timeout or 30.0)
        if line is None:
            raise TimeoutError("sandbox response timed out")
        if not line:
            raise RuntimeError("sandbox closed unexpectedly")
        message = json.loads(line)
        if message.get("type") == "error":
            raise RuntimeError(message.get("error", "sandbox error"))
        return {k: v for k, v in message.items() if k != "type"}

    def close(self) -> dict[str, Any]:
        if self.closed:
            return {"wall_seconds": round(time.perf_counter() - self.started, 4)}
        self.closed = True
        try:
            if self.process.stdin and self.process.poll() is None:
                self.process.stdin.write(json.dumps({"type": "stop"}) + "\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except OSError:
                pass
        stderr = ""
        if self.process.stderr:
            try:
                stderr = self.process.stderr.read()[-4000:]
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(self.rootfs, ignore_errors=True)
        return {
            "wall_seconds": round(time.perf_counter() - self.started, 4),
            "outer_log": stderr,
        }

    def __enter__(self) -> "InteractiveSandbox":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def execute_scripts(
    scripts: list[str],
    *,
    command_timeout: float = 20.0,
    overall_timeout: float = 120.0,
) -> dict[str, Any]:
    """Run scripts sequentially in one fresh namespace/root; return all results."""
    sandbox: InteractiveSandbox | None = None
    executions: list[dict[str, Any]] = []
    error: str | None = None
    started = time.perf_counter()
    try:
        sandbox = InteractiveSandbox(command_timeout=command_timeout)
        for script in scripts:
            remaining = overall_timeout - (time.perf_counter() - started)
            if remaining <= 0:
                error = "overall_timeout"
                break
            executions.append(sandbox.execute(script, timeout=remaining))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        meta = sandbox.close() if sandbox else {"wall_seconds": round(time.perf_counter() - started, 4)}
    return {
        "executions": executions,
        "sandbox_error": error,
        **meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-server", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inner_server:
        inner_server(args.inner_server)
    else:
        parser.error("use the Python API or pass --inner-server")


if __name__ == "__main__":
    main()
