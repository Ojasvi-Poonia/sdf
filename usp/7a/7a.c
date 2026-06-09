#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>

jmp_buf jb;

int gv;
volatile int vv = 4;

void f2()
{
    longjmp(jb, 1);
}

void f1(int av, int rv, int sv)
{
    printf("In f1()\n");
    printf("gv=%d av=%d rv=%d vv=%d sv=%d\n",
           gv, av, rv, vv, sv);

    f2();
}

int main()
{
    int av = 2;
    register int rv = 3;
    static int sv = 5;

    gv = 1;

    if(setjmp(jb) != 0)
    {
        printf("After longjmp\n");
        printf("gv=%d av=%d rv=%d vv=%d sv=%d\n",
               gv, av, rv, vv, sv);
        return 0;
    }

    gv = 95;
    av = 96;
    rv = 97;
    vv = 98;
    sv = 99;

    f1(av, rv, sv);

    return 0;
}
