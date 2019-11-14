#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>

using namespace std;

pair<int, vector<int>> activitySelection(const vector<int> &s, const vector<int> &f, int n){
    vector<int> p(n);
    for(int i = 0; i < n; ++i){
        p[i] = i;
    }

    sort(p.begin(), p.end(), [&](int a, int b){
        return f[a] < f[b];
    });

    vector<int> r;

    int c = 1;
    r.push_back(p[0] + 1);
    
    int k = 0;
    
    for(int i = 1; i < n; ++i){
        if(s[p[i]] >= f[p[k]]){
            c++;
            r.push_back(p[i] + 1);
            k = i;
        }
    }

    return make_pair(c, r);
}

int main(){
    int n;
    cin >> n;

    vector<int> s(n), f(n);

    for(auto &el : s){
        cin >> el;
    }

    for(auto &el : f){
        cin >> el;
    }

    int c;
    vector<int> r;
    tie(c, r) = activitySelection(s, f, n);

    cout << c << endl;
    for(auto &el : r){
        cout << el << ' ';
    }
    cout << endl;
}
