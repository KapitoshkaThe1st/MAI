#include <iostream>

#include <cmath>
#include <vector>

using namespace std;

double pi = 3.14159265358979323846;

double eps = 0.001;

int inf = 100;

// double a = sqrt(E / ro);
double a = 1000;

double l = 0.1;

double mu1(double t){
    return sin(t);
}

double f(double, double t){
    return sin(t);
}

template<class F>
double integral(const F &f, double a, double b, double h=eps){
    double w = b-a;
    int n = ceil(w / h);
    if (n == 0)
        return w * f(a + w/2.0);

    double hh = w / n;
    
    double s = f(a) * 0.5;
    for (int i = 0; i < n; ++i){
        s += f(a + hh*i);
    }
    s += f(b) * 0.5;

    return s * hh;
}

template<class F>
double series(const F &f, int a, int b){
    double s = 0;
    for (int k = a; k <= b; ++k){
        s += f(k);
    }
    return s;
}

template<class F>
double deriv(const F &f, double x, double h=eps){
    double fph = f(x + h);
    double fmh = f(x - h);

    return (fph - fmh) / (2.0 * h);
}

template<class F>
double deriv2(const F &f, double x, double h=eps){
    return (f(x + h) - 2.0*f(x) + f(x - h)) / (h*h);
}

double g1(double){
    return -mu1(0.0);
}

double g2(double){
    return -deriv(mu1, 0.0);
}

double int_g1(int k){
    auto ff = [k](double xi) {return g1(xi) * sin((2.0*k+1.0)/(2.0*l)*pi * xi);};
    return integral(ff, 0.0, l);
}

double int_g2(int k){
    auto ff = [k](double xi) {return g2(xi) * sin((2.0*k+1.0)/(2.0*l)*pi * xi);};
    return integral(ff, 0.0, l);
}

double v(double x, double t){
    auto ff = [t, x](int k) {return (4.0 / ((2.0*k+1)*pi*a) * int_g2(k) * sin((2.0*k+1.0)/(2.0*l)*pi*a*t) + 2.0/l * int_g1(k) * cos((2.0*k+1.0)/(2.0*l)*pi*a*t)) * sin((2.0*k+1.0)/(2.0*l)*pi*x);};
    return series(ff, 0, inf);
}

double int_f(double tau, int k){
    auto ff = [k, tau](double xi) {return f(xi, tau) * sin((2.0*k+1.0)/(2.0*l)*pi * xi);};
    return integral(ff, 0.0, l);
}

double int_int(double t, int k){
    auto ff = [k, t](double tau) {return int_f(tau, k) * sin((2.0*k+1.0)/(2.0*l)*pi*a*(t-tau));};
    return integral(ff, 0.0, t);
}

double w(double x, double t){
    auto ff = [t, x](int k) {return int_int(t, k) / (2.0*k+1.0) * sin((2.0*k+1.0)/(2.0*l)*pi * x);};
    return 4 / (a*pi) * series(ff, 0, inf);
}

double u1(double, double t){
    return mu1(t);
}

double u(double x, double t){
    double u1v = u1(x, t);
    double vv = v(x, t);
    double wv = w(x, t);

    // cout << "x: " << x << " t: " << t << endl;
    // cout << "u1: " << u1v << " v: " << vv << " w: " << wv << '\n';
    return u1v + vv + wv;
}

vector<double> linspace(double a, double b, int n=50){
    double w = (b-a) / (n-1);

    vector<double> res(n);
    for(int i = 0; i < n; ++i)
        res[i] = a+i*w;
    return res;
}

template<class F>
vector<double> compute(const F &f, const vector<double> &v){
    size_t n = v.size();
    vector<double> res(n);
    for(size_t i = 0; i < n; ++i)
        res[i] = f(v[i]);
    
    return res;
}

void print_vector(const vector<double> &v){
    for(auto el : v)
        cout << el << '\n';
}

int main(){
    cout.precision(17);
    int k = 8;
    int n = 50;
    cout << n << endl;

    vector<double> xvals = linspace(0, l, n);

    print_vector(xvals);

    cout << k << endl;
    for(int i = 0; i < k; ++i){
        double t = 0.00003*i;
        // double t = 0.5*i;
        cout.precision(5);
        cout << t << endl;
        cout.precision(17);
        vector<double> yvals = compute([t](double x){return u(x, t);}, xvals);
        print_vector(yvals);
    }

    // cout << "check:" << endl;

    // vector<double> y = compute([](double x){
    //     return w(x, 0);}, xvals);
    // print_vector(y);

    // double x = l/2;

    // double ve = v(x, eps);
    // double vz = v(x, 0);

    // cout << (ve-vz) / eps << endl;

    // double max_t = 10;

    // vector<double> tvals = linspace(0, max_t, n);
    // cout << "tvals:" << endl;
    // print_vector(tvals);

    // vector<double> y = compute([](double t){
    //     double wl = w(l, t);
    //     double wle = w(l, t-eps);

    //     return (wl - wle) / eps;}, tvals);
    // cout << "y:" << endl;
    // print_vector(y);
}