/*  KAV2019   */
#include "mlisp.h"

double var1 = 1;
double var2 = 2;
double var1000000 = 1000000;
double var10000 = 10000;
double dd = 29;
double mm = 4;
double yyyy = 1999;

double even__bits(double n);
double odd__bits(double n);
double display__bin(double n);
double report__results(double n);
//________________ 
double even__bits(double n){
    return (((n == 0) ? var1
        : ((remainder(n, var2) == 0) ? even__bits(quotient(n, var2))
            : (true ? odd__bits(quotient(n, var2))
                : _infinity))));
}

double odd__bits(double n){
    return (((n == 0) ? 0
        : ((remainder(n, var2) == 0) ? odd__bits(quotient(n, var2))
            : (true ? even__bits(quotient(n, var2))
                : _infinity))));
}

double display__bin(double n){
    display(remainder(n, var2));
    return (((n == 0) ? 0
        : (true ? display__bin(quotient(n, var2))
            : _infinity)));
}

double report__results(double n){
    display("Happy birthday to you!\n\t");
    display(n);
    display(" (decimal)\n\t");
    display__bin(n);
    display("(reversed binary)\n");
    display("\teven?\t");
    display(((even__bits(n) == var1) ? "yes" : "no"));
    newline();
    display("\todd?\t");
    display(((odd__bits(n) == var1) ? "yes" : "no"));
    newline();
    return 0;
}

int main(){
    display(report__results(((dd * var1000000) + (mm * var10000) + yyyy))); newline();
    std::cin.get();
    return 0;
}

