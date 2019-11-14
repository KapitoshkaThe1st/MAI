//coin19.cpp 2019
#include "mlisp.h"

double VARIANT = 9;
double LAST__DIGIT__OF__GROUP__NUMBER = 8; //6,8
double LARGEST__COIN = 20;

bool implication_Q(bool x_Q, bool y_Q);
double cc(double amount, double largest__coin);
double count__change(double amount);
double next__coin(double coin);
double GR__AMOUNT();

bool implication_Q(bool x_Q, bool y_Q){
    return !(x_Q && !y_Q);
}

double cc(double amount, double largest__coin){
    return ((amount == 0 || largest__coin == 1) ? 1
    : implication_Q(amount >= 0, largest__coin == 0) ? 0
    : (cc(amount, next__coin(largest__coin)) +
    cc(amount - largest__coin, largest__coin)));
}

double count__change(double amount){
    return cc(amount, LARGEST__COIN);
}

double next__coin(double coin){
    return (coin == 20 ? 15
    : coin == 15 ? 10
    : coin == 10 ? 5
    : coin == 5 ? 3
    : coin == 3 ? 2
    : 1);
}

double GR__AMOUNT(){
    return remainder(100 * LAST__DIGIT__OF__GROUP__NUMBER + VARIANT, 137);
}

int main(){
    display(" KAV variant ");
    display(VARIANT); newline();
    display(" 1-2-3-5-10-15-20"); newline();
    display("count__change for 100 \t= ");
    display(count__change(100)); newline();
    display("count__change for ");
    display(GR__AMOUNT());
    display(" \t= ");
    display(count__change(GR__AMOUNT())); newline();

    std::cin.get();
    return 0;
}
