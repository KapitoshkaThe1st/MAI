#include <iostream>

#include <cmath>
#include <vector>

using namespace std;

double pi = 3.14159265358979323846;

double eps = 0.001;

int inf = 100;

double l = 0.1;
double r = 0.005;
double K = 60;
double S = 2 * pi * r;

double alpha1 = 0.1;

double a_sqr = 1e-6;
// double b_sqr = 2.0*alpha1*a_sqr/(K*r);
double b_sqr = 1e-6;


// double a_sqr = 0.001;
// double b_sqr = 0;

double u_env = 1000;
double T0 = 300;

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

double mu1(double t){
    return 300.0 * exp(-t);
}

double mu2(double t){
    return 500.0 * exp(-t);
    // return 0;
}

double g1(double t){
    return mu1(t);
}

double g2(double t){
    return mu2(t) / (K * S);
}

double psi2(double t){
    return exp(b_sqr * t) * g2(t);
}

double psi1(double t){
    return exp(b_sqr * t) * g1(t);
}

double u3(double x, double t){
    return psi2(t)*x+psi1(t);
}

double phi(double){
    return T0;
}

double phi1(double x){
    return phi(x) - u3(x, 0);
}

double int_phi1(int k){
    auto ff = [k](double xi){return phi1(xi) * sin((2.0*k+1.0)*pi / (2.0*l) * xi);};
    return integral(ff, 0.0, l);
}

double u1(double x, double t){
    auto ff = [x, t](int k){return int_phi1(k) * exp(-a_sqr * pow((2.0*k+1.0)*pi / (2.0*l), 2) * t) * sin((2.0*k+1.0)*pi / (2.0*l) * x);};
    return 2.0 / l * series(ff, 0, inf);
}

double f(double, double){
    return b_sqr * u_env;
}

double f1(double x, double t){
    return f(x, t) * exp(b_sqr * t);
}

double f2(double x, double t){
    return f1(x,t) - deriv([x](double t){return u3(x, t);}, t);
}

double int_f2(double tau, int k){
    auto ff = [tau, k](double xi){return f2(xi, tau) * sin((2.0*k+1.0)*pi / (2.0*l) * xi);};
    return integral(ff, 0, l);
}

double int_int(double t, int k){
    auto ff = [t, k](double tau){return int_f2(tau, k) * exp(-a_sqr * pow((2.0*k+1.0)*pi / (2.0*l), 2) * (t-tau));};
    return integral(ff, 0.0, t);
}

double u2(double x, double t){
    auto ff = [x, t](int k){return int_int(t, k) * sin((2.0*k+1.0)*pi / (2.0*l) * x);};
    return 2.0 / l * series(ff, 0, inf);
}

double v(double x, double t){
    double u1v = u1(x, t);
    double u2v = u2(x, t);
    double u3v = u3(x, t);

    cerr << "u1: " << u1v << " u2: " << u2v << " u3: " << u3v << '\n';

    return u1v + u2v + u3v;
}

double u(double x, double t){
    return exp(-b_sqr * t) * v(x,t);
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
    cerr << "a_sqr: " << a_sqr << endl;
    cerr << "b_sqr: " << b_sqr << endl;

    cout.precision(17);
    int k = 4;
    int n = 150;
    cout << n << endl;

    vector<double> xvals = linspace(0, l, n);

    print_vector(xvals);

    cout << k << endl;
    for(int i = 0; i < k; ++i){
        // double t = 0.00003*i;
        double t = 0.5 * i;
        cout.precision(5);
        cout << t << endl;
        cout.precision(17);
        vector<double> yvals = compute([t](double x){return u(x, t);}, xvals);
        print_vector(yvals);
    }

    // cout << "check:" << endl;

    // vector<double> y = compute([](double x){
    //     return u1(x, 0) - phi1(x);}, xvals);
    // print_vector(y);

    // double x = l/2;

    // double ve = v(x, eps);
    // double vz = v(x, 0);

    // cout << (ve-vz) / eps << endl;

    // double max_t = 4;

    // vector<double> tvals = linspace(0, max_t, n);
    // cout << "tvals:" << endl;
    // print_vector(tvals);

    // vector<double> y = compute([](double t){
    //     double wl = u3(l+eps, t);
    //     double wle = u3(l-eps, t);

    //     return (wl - wle) / (2*eps) - psi2(t);}, tvals);
    // cout << "y:" << endl;
    // print_vector(y);

    // vector<double> y = compute([](double t){
    //     return u(0,t)-mu1(t);}, tvals);
    // cout << "y:" << endl;
    // print_vector(y);
}