#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

int main(int argc, char *argv[])
{
    DIR *dp = opendir(argv[1]);
    struct dirent *d;
    struct stat s;

    while((d = readdir(dp)) != NULL)
    {
        stat(d->d_name, &s);

        printf("%ld %o %d %d %s %s\n",
               s.st_ino,
               s.st_mode,
               s.st_uid,
               s.st_gid,
               ctime(&s.st_atime),
               d->d_name);
    }

    closedir(dp);

    return 0;
}
