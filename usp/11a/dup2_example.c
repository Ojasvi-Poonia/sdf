#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main()
{
    int fd1, fd2;

    fd1 = open("t12.txt", O_RDWR);

    fd2 = dup2(fd1, 5);

    printf("fd1 = %d\n", fd1);
    printf("fd2 = %d\n", fd2);

    write(fd1, "abcdef", 6);

    return 0;
}
