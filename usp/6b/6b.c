#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int fd;
    char buf[100];
    struct flock lock;

    fd = open(argv[1], O_RDWR);

    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_END;
    lock.l_start = -100;
    lock.l_len = 100;

    printf("Press Enter to Lock\n");
    getchar();

    if(fcntl(fd, F_SETLK, &lock) == -1)
    {
        fcntl(fd, F_GETLK, &lock);
        printf("Locked by PID: %d\n", lock.l_pid);
        return 0;
    }

    printf("File Locked\n");

    lseek(fd, -50, SEEK_END);

    read(fd, buf, 50);

    buf[50] = '\0';

    printf("Last 50 Bytes:\n%s\n", buf);

    printf("Press Enter to Unlock\n");
    getchar();

    lock.l_type = F_UNLCK;

    fcntl(fd, F_SETLK, &lock);

    printf("Unlocked\n");

    close(fd);

    return 0;
}
