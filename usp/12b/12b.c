#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main()
{
    int pid;

    char *env[] =
    {
        "USER=unknown",
        "PATH=/tmp",
        NULL
    };

    pid = fork();

    if(pid == 0)
    {
        execle("./echoall",
               "echoall",
               "myarg1",
               "myarg2",
               NULL,
               env);
    }

    wait(NULL);

    execlp("./echoall",
           "echoall",
           "only1arg",
           NULL);

    return 0;
}
