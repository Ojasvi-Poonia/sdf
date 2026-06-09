#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main()
{
    int pid, pid2;

    pid = fork();

    if(pid == 0)
    {
        int pid3 = fork();

        if(pid3 == 0)
        {
            sleep(5);

            printf("Second Child PID = %d\n", getpid());
            printf("Parent PID = %d\n", getppid());

            exit(0);
        }
        else
        {
            printf("First Child PID = %d\n", getpid());
            exit(0);
        }
    }

    pid2 = waitpid(pid, NULL, 0);

    printf("Terminated Child PID = %d\n", pid2);

    return 0;
}
