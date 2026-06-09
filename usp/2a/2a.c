#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    struct stat s;

    stat(argv[1], &s);

    printf("File Name: %s\n", argv[1]);
    printf("Size: %ld bytes\n", s.st_size);
    printf("Permissions: %o\n", s.st_mode & 0777);
    printf("Links: %ld\n", s.st_nlink);
    printf("UID: %d\n", s.st_uid);
    printf("GID: %d\n", s.st_gid);

    return 0;
}
