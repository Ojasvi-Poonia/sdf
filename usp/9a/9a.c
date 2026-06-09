#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main()
{
    mode_t oldmask;

    oldmask = umask(0022);

    printf("Old umask = %03o\n", oldmask);

    int fd = open("t1.txt", O_CREAT | O_WRONLY, 0777);

    close(fd);

    chmod("t1.txt", 0644);

    printf("Permissions changed to 0644\n");

    return 0;
}
