#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

int main(int argc, char *argv[])
{
    DIR *dp = opendir(argv[1]);
    struct dirent *entry;

    while((entry = readdir(dp)) != NULL)
    {
        if(strcmp(entry->d_name,".") == 0 || strcmp(entry->d_name,"..") == 0)
            continue;

        char path[100];

        sprintf(path,"%s/%s",argv[1],entry->d_name);

        int fd = open(path,O_RDONLY);

        int size = lseek(fd,0,SEEK_END);

        if(size == 0)
        {
            unlink(path);
            printf("Removed: %s\n",entry->d_name);
        }

        close(fd);
    }

    closedir(dp);

    return 0;
}
