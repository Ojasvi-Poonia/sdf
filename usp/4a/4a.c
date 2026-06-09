#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    if(argc == 3)
    {
        link(argv[1], argv[2]);
        printf("Hard Link Created\n");
    }
    else if(argc == 4)
    {
        link(argv[1], argv[2]);
        symlink(argv[2], argv[3]);
        printf("Soft Link Created\n");
    }

    return 0;
}
