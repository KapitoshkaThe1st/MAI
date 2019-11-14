/*  KAV2019   */
#include "mlisp.h"

double VARIANT = 9;
double LAST__DIGIT__OF__GROUP__NUMBER = 8;
double LARGEST__COIN = 20;
double var1 = 1;
double var2 = 2;
double var3 = 3;
double var5 = 5;
double var10 = 10;
double var15 = 15;
double var20 = 20;
double var100 = 100;
double var137 = 137;

bool my__not_Q(bool x_Q);
bool my__or_Q(bool x_Q, bool y_Q);
bool implication_Q(bool x_Q, bool y_Q);
double cc(double amount, double largest__coin);
double count__change(double amount);
double next__coin(double coin);
double GR__AMOUNT();
//________________ 
bool my__not_Q(bool x_Q){
    return (0 == ((x_Q ? e
        : (true ? 0
            : _infinity))));
}

bool my__or_Q(bool x_Q, bool y_Q){
    return (e == ((x_Q ? e
        : (true ? ((y_Q ? e
                    : (true ? 0
                        : _infinity)))
            : _infinity))));
}

bool implication_Q(bool x_Q, bool y_Q){
    return my__or_Q(my__not_Q(x_Q), y_Q);
}

double cc(double amount, double largest__coin){
    return ((my__or_Q((amount == 0), (largest__coin == var1)) ? var1
        : (implication_Q(my__not_Q((amount < 0)), (largest__coin == 0)) ? 0
            : (true ? (cc(amount, next__coin(largest__coin)) + cc((amount - largest__coin), largest__coin))
                : _infinity))));
}

double count__change(double amount){
    return cc(amount, LARGEST__COIN);
}

double next__coin(double coin){
    return (((coin == var20) ? var15
        : ((coin == var15) ? var10
            : ((coin == var10) ? var5
                : ((coin == var5) ? var3
                    : ((coin == var3) ? var2
                        : (true ? var1
                            : _infinity)))))));
}

double GR__AMOUNT(){
    return remainder(((var100 * LAST__DIGIT__OF__GROUP__NUMBER) + VARIANT), var137);
}

int main(){
    display(" KAV variant ");
    display(VARIANT);
    newline();
    display(" 1-2-3-5-10-15-20");
    newline();
    display("count__change for 100 \t= ");
    display(count__change(var100));
    newline();
    display("count__change for ");
    display(GR__AMOUNT());
    display(" \t= ");
    display(count__change(GR__AMOUNT()));
    newline();
    std::cin.get();
    return 0;
}

