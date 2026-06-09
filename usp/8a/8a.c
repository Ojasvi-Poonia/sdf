#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    struct stat buf;
    char *ptr;

    for(int i = 1; i < argc; i++)
    {
        lstat(argv[i], &buf);

        if(S_ISREG(buf.st_mode))
            ptr = "Regular File";
        else if(S_ISDIR(buf.st_mode))
            ptr = "Directory";
        else if(S_ISCHR(buf.st_mode))
            ptr = "Character Special File";
        else if(S_ISBLK(buf.st_mode))
            ptr = "Block Special File";
        else if(S_ISFIFO(buf.st_mode))
            ptr = "FIFO";
        else if(S_ISLNK(buf.st_mode))
            ptr = "Symbolic Link";
        else if(S_ISSOCK(buf.st_mode))
            ptr = "Socket";
        else
            ptr = "Unknown";

        printf("%s : %s\n", argv[i], ptr);
    }

    return 0;
}
