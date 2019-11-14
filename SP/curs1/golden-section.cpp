/*  KAV2019   */
#include "mlisp.h"

double var05 = 0.5;
double var2 = 2;
double var3 = 3;
double var4 = 4;
double var5 = 5;
double var7 = 7;
double var109 = 109;
double var110 = 110;
double a = 0;
double b = 2;
double z = 0;
double eps = 0.00001;
double mphi = 0;
double xmin = 0;

double fun(double x);
double golden__section__search(double a, double b);
double golden__start(double a, double b);
double __KAV2019__try(double a, double b, double xa, double ya, double xb, double yb);
//________________ 
double fun(double x){
    x = (x - (double(var109) / var110 / e));
    z = x;
    return ((var5 * expt(log(expt(atan((z - var2)), var2)), var4)) - z - var7);
}

double golden__section__search(double a, double b){
    {
        double xmin((((a < b) ? golden__start(a, b)
            : (true ? golden__start(b, a)
                : _infinity))));
        newline();
        return xmin;
    }
}

double golden__start(double a, double b){
    mphi = (var05 * (var3 - sqrt(var5)));
    {
        double xa((a + (mphi * (b - a)))),
        xb((b - (mphi * (b - a))));
        return __KAV2019__try(a, b, xa, fun(xa), xb, fun(xb));
    }
}

double __KAV2019__try(double a, double b, double xa, double ya, double xb, double yb){
    return (((abs((a - b)) < eps) ? ((a + b) * var05)
        : (true ? display("+"),
                (((ya < yb) ? b = xb,
                        xb = xa,
                        yb = ya,
                        xa = (a + (mphi * (b - a))),
                        __KAV2019__try(a, b, xa, fun(xa), xb, yb)
                    : (true ? a = xa,
                            xa = xb,
                            ya = yb,
                            xb = (b - (mphi * (b - a))),
                            __KAV2019__try(a, b, xa, ya, xb, fun(xb))
                        : _infinity)))
            : _infinity)));
}

int main(){
    xmin = golden__section__search(a, b);
    display("interval=\t[");
    display(a);
    display(" , ");
    display(b);
    display("]\n");
    display("xmin=\t\t");
    display(xmin); newline();
    display("f(xmin)=\t");
    display(fun(xmin)); newline();
    std::cin.get();
    return 0;
}

