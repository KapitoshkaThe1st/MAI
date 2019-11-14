/*  xxx2019   */
#include "mlisp.h"
double __xxx2019__try(double x);
double f(double x, double y);
//________________ 
double __xxx2019__try(double x){
	x = f(x, x);
	return x;
}
double f(double x, double y){
	return x * y;
}
int main(){
	display(__xxx2019__try(3)); newline();
	std::cin.get();
	return 0;
}

