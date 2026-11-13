# Vendored sandbox helpers

- `faked-sysv` and `libfakeroot-sysv.so`: Debian 13 (`fakeroot` 1.37.1.1-1, `libfakeroot` 1.37.1.1-1), GPL-3-or-later. See `GPL-3.txt`. Upstream project: https://salsa.debian.org/clint/fakeroot
- `force_chown_eperm.c` / `libforce_chown_eperm.so`: small local shim used only inside the sandbox to make unprivileged `chown` fail with `EPERM` rather than `EINVAL`; `libfakeroot` then records the intended fake owner. The source is included and can be rebuilt with:

  ```bash
  gcc -shared -fPIC -O2 -o libforce_chown_eperm.so force_chown_eperm.c
  ```

The prebuilt `.so` files are Linux x86-64/glibc binaries. On another architecture, rebuild the shim and replace the Debian fakeroot files with your distribution's equivalents.
