#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[])
{
    int pid = fork();

    if(pid == 0)
    {
        execl("./p23", "p23", argv[1], argv[2], NULL);
    }
    else
    {
        wait(NULL);
        printf("Child Process Completed\n");
    }

    return 0;
}
