#include <iostream>

int main(void)
{
   int foo = 10, bar = 15;
   __asm__ __volatile__("addl  %%ebx,%%eax"
                        :"=a"(foo)
                        :"a"(foo), "b"(bar)
                        );
   //printf("foo+bar=%d\n", foo);
   std::cout << "res: " << foo << "\n";
   return 0;
}
