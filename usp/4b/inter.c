#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main()
{
    int pid = fork();

    if(pid == 0)
    {
        execl("./textinterpreter",
              "textinterpreter",
              "myarg1",
              "myarg2",
              "myarg3",
              NULL);
    }
    else
    {
        wait(NULL);
    }

    return 0;
}
