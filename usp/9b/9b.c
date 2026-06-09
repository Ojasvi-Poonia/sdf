#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int fd;
    char buf[21];

    fd = open(argv[1], O_RDONLY);

    read(fd, buf, 20);
    buf[20] = '\0';
    printf("First 20 Characters : %s\n", buf);

    lseek(fd, 10, SEEK_SET);
    read(fd, buf, 20);
    buf[20] = '\0';
    printf("20 Characters From 10th Byte : %s\n", buf);

    lseek(fd, 10, SEEK_CUR);
    read(fd, buf, 20);
    buf[20] = '\0';
    printf("20 Characters From Current Offset : %s\n", buf);

    int size = lseek(fd, 0, SEEK_END);

    printf("File Size = %d Bytes\n", size);

    close(fd);

    return 0;
}
