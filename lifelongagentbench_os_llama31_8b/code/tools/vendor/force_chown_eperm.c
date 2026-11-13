/* Force chown-family syscalls to fail with EPERM inside the sandbox.
 *
 * With only uid/gid 0 mapped in a user namespace, chown to a newly created
 * user returns EINVAL ("Invalid argument").  libfakeroot only fakes ownership
 * when the underlying call fails with EPERM.  Loading this shim *after*
 * libfakeroot (LD_PRELOAD="libfakeroot-sysv.so libforce_chown_eperm.so")
 * makes its RTLD_NEXT calls see EPERM and store the intended fake owner.
 *
 * Build: gcc -shared -fPIC -o libforce_chown_eperm.so force_chown_eperm.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

int chown(const char *path, uid_t owner, gid_t group) {
    (void)path; (void)owner; (void)group;
    errno = EPERM;
    return -1;
}

int lchown(const char *path, uid_t owner, gid_t group) {
    (void)path; (void)owner; (void)group;
    errno = EPERM;
    return -1;
}

int fchown(int fd, uid_t owner, gid_t group) {
    (void)fd; (void)owner; (void)group;
    errno = EPERM;
    return -1;
}

int fchownat(int dirfd, const char *path, uid_t owner, gid_t group, int flags) {
    (void)dirfd; (void)path; (void)owner; (void)group; (void)flags;
    errno = EPERM;
    return -1;
}
