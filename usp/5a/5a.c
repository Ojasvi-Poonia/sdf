#include <stdio.h>
#include <sys/stat.h>
#include <utime.h>

int main(int argc, char *argv[])
{
    struct stat s;
    struct utimbuf t;

    stat(argv[1], &s);

    t.actime = s.st_atime;
    t.modtime = s.st_mtime;

    utime(argv[2], &t);

    printf("Access and Modification Time Copied\n");

    return 0;
}
