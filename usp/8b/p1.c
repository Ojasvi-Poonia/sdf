#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    if(access(argv[1], F_OK) == 0)
    {
        printf("File %s exists\n", argv[1]);
    }
    else
    {
        printf("File %s does not exist\n", argv[1]);
    }

    return 0;
}
