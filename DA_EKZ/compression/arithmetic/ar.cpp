#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <tuple>
#include <cassert>

using namespace std;

pair<vector<double>, vector<double>> arithm_stat(const string &str){
    vector<int> dict(256);

    int n = str.length();
    for (int i = 0; i < n; ++i) {
        dict[str[i]]++;
    }

    vector<double> l(256);
    vector<double> h(256);

    double ll = 0.0;
    double hh = 0.0;

    for (int i = 0; i < 256; ++i) {
        if (dict[i] > 0) {
            hh = ll + (double)dict[i] / n;
            l[i] = ll;
            h[i] = hh;
            ll = hh;
        }
    }

    for (int c = 0; c < 256; ++c) {
        if (dict[c] > 0) {
            cout << (char)c << ": [" << l[c] << ", " << h[c] << ")" << endl;
        }
    }

    return make_pair(l, h);
}

double arithm_encode(const string &str){
    vector<double> l, h;
    tie(l, h) = arithm_stat(str);

    double ll = 0.0;
    double hh = 1.0;
    for(char c : str){
        double temp_l = ll + (hh - ll) * l[c];
        double temp_h = ll + (hh - ll) * h[c];
        ll = temp_l;
        hh = temp_h;

        cout << "l: " << ll << endl;
        // cout << "h: " << hh << endl;
    }

    // return (ll + hh) / 2;
    return ll;
}

string arithm_decode(vector<double> &l, vector<double> &h, double code){
    string str;
    while(code > 0){
        int i = 0;
        while (h[i] < code) {
            i++;
        }
        assert(code < h[i] && code > l[i]);
        str.push_back((char)i);
        cout << l[i] << " - " << h[i] << endl;
        cout << (char)i << endl;
        cout << code << endl;
        code = (code - l[i]) / (h[i] - l[i]);
    }
    return str;
}

int main(){
    string s;
    cin >> s;

    cout << s << endl;
    double code = arithm_encode(s); 
    cout << code << endl;

    vector<double> l, h;
    tie(l, h) = arithm_stat(s);

    cout << arithm_decode(l, h, code) << endl;
}