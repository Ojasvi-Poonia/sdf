#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int my_system(char *cmd)
{
    int status;

    if(fork() == 0)
    {
        execl("/bin/sh", "sh", "-c", cmd, NULL);
        exit(1);
    }
    else
    {
        wait(&status);
    }

    return status;
}

int main()
{
    printf("Executing ls -l\n");

    my_system("ls -l");

    return 0;
}
