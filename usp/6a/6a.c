#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main()
{
    int fd1, fd2;
    char buf[20];

    fd1 = open("example.txt", O_RDWR);
    fd2 = open("sample.txt", O_CREAT | O_RDWR, 0777);

    fd2 = dup2(fd1, fd2);

    read(fd1, buf, 20);

    lseek(fd2, 0, SEEK_END);

    write(fd2, buf, 20);

    printf("%s\n", buf);

    close(fd1);
    close(fd2);

    return 0;
}
