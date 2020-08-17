#include <iostream>
#include <string>

#include "../../include/integral.h"

using namespace std;

double func(double x){
    return 1.0 / (x*x + 4.0);
}

int main(){

    string mode;
    cin >> mode;

    double a, b;
    cin >> a >> b;

    double h;
    cin >> h;

    double hh = h * 0.5;

    if(mode == "rectangle"){
        cout << "rectangle method" << endl;
        double s1 = integral_rectangle(func, a, b, h);
        double s2 = integral_rectangle(func, a, b, hh);

        cout << "with h: " << h << ": " << s1 << endl;
        cout << "with h: " << hh << ": " << s2 << endl;

        cout << "error estimation: " << error_estimation(s1, s2, IntegrationMethod::Rectangle) << endl;
    }
    else if(mode == "trapezoid"){
        cout << "trapezoid method" << endl;
        double s1 = integral_trapezoid(func, a, b, h);
        double s2 = integral_trapezoid(func, a, b, hh);

        cout << "with h: " << h << ": " << s1 << endl;
        cout << "with h: " << hh << ": " << s2 << endl;

        cout << "error estimation: " << error_estimation(s1, s2, IntegrationMethod::Trapezoid) << endl;
    }
    else if(mode == "simpson"){
        cout << "Simpson method" << endl;
        double s1 = integral_simpson(func, a, b, h);
        double s2 = integral_simpson(func, a, b, hh);

        cout << "with h: " << h << ": " << s1 << endl;
        cout << "with h: " << hh << ": " << s2 << endl;

        cout << "error estimation: " << error_estimation(s1, s2, IntegrationMethod::Simpson) << endl;
    }
    else{
        cout << "There is no \"" << mode << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }
}
