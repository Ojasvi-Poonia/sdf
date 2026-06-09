#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
int fd = open(argv[1], O_RDONLY);
int size = lseek(fd, 0, SEEK_END);

char ch;

for(int i = 1; i <= size; i++)
{
    lseek(fd, -i, SEEK_END);
    read(fd, &ch, 1);
    printf("%c", ch);
}

close(fd);

return 0;
}
