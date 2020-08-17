#include <iostream>

#include "../../include/tdma.h"
#include "../../include/matrix.h"

using namespace std;

using vec = vector<double>;

int main(){
    int n;
    cin >> n;

    vec a(n-1), b(n), c(n-1), d(n);

    cout << "--- INPUT ---\n";

    for(double &i : a)
        cin >> i;

    for(double &i : b)
        cin >> i;
    
    for(double &i : c)
        cin >> i;

    for(double &i : d)
        cin >> i;

    cout << "a: ";
    cout << a << endl;
    cout << "b: ";
    cout << b << endl;
    cout << "c: ";
    cout << c << endl;
    cout << "d: ";
    cout << d << endl;

    vec x = tdma(a, b, c, d, n);

    cout << "\n--- SOLUTION ---\n";
    cout << "x: ";
    for(double i : x)
        cout << i << ' ';
    cout << endl;
}
