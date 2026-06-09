#include <stdio.h>
#include <signal.h>
#include <unistd.h>

void handler(int sig)
{
    printf("\nCaught SIGINT %d\n", sig);

    struct sigaction sa;

    sa.sa_handler = SIG_DFL;
    sa.sa_flags = 0;

    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, NULL);
}

int main()
{
    struct sigaction sa;

    sa.sa_handler = handler;
    sa.sa_flags = 0;

    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, NULL);

    while(1)
    {
        printf("Press Ctrl+C\n");
        pause();
    }

    return 0;
}
