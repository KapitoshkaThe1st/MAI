#include "mlisp.h"
//...
double odd__bits(double n)
 {
  return (n == 0 ? 0
       : remainder(n,2) == 0 ?
                odd__bits (quotient(n,2))
       : true ? even__bits(quotient(n,2))
       : _infinity);
 }
//...
int main(){
 display(report__results (dd*1000000+
                         mm*10000+
                         yyyy));
 newline();

 std::cin.get();
 return 0;
}
