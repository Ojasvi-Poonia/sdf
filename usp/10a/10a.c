#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <fcntl.h>
#include <sys/stat.h>

void create_daemon()
{
    int pid = fork();

    if(pid > 0)
        exit(0);

    setsid();

    umask(0);

    chdir("/");

    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
}

int main()
{
    create_daemon();

    openlog("daemon", LOG_PID, LOG_DAEMON);

    while(1)
    {
        syslog(LOG_NOTICE, "Daemon Running");
        sleep(30);
    }

    closelog();

    return 0;
}
