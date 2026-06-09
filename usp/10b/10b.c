#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>

int main()
{
    int status;

    int pid1 = fork();

    if(pid1 == 0)
    {
        printf("First Child PID = %d\n", getpid());
        sleep(2);
        exit(0);
    }

    int pid2 = fork();

    if(pid2 == 0)
    {
        printf("Second Child PID = %d\n", getpid());
        sleep(4);
        exit(0);
    }

    wait(&status);
    printf("First wait completed\n");

    waitpid(pid2, &status, 0);
    printf("Second waitpid completed\n");

    return 0;
}
