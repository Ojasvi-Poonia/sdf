#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);

    int sum = a + b;

    printf("Sum = %d\n", sum);

    return 0;
}
