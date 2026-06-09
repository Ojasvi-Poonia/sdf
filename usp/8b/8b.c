#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main()
{
    int pid = fork();

    if(pid == 0)
    {
        printf("Child PID = %d\n", getpid());

        execl("./p1", "p1", "example.txt", NULL);
    }
    else
    {
        printf("Parent PID = %d\n", getpid());

        wait(NULL);

        printf("Child Process Finished\n");
    }

    return 0;
}
