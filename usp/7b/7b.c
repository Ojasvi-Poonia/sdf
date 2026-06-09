#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int src, dest;
    char buf[1024];
    int n;

    src = open(argv[1], O_RDONLY);

    dest = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0777);

    while((n = read(src, buf, sizeof(buf))) > 0)
    {
        write(dest, buf, n);
    }

    close(src);
    close(dest);

    return 0;
}
