#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> lis(const vector<int> a){
    int n = a.size();

    vector<int> d(n), p(n);

    d[0] = 1;
    p[0] = -1;
    for(int i = 1; i < n; ++i){
        int maxInd = 0;
        for(int j = 1; j < i; ++j){
            if(a[j] < a[i] && d[j] > d[maxInd]){
                maxInd = j;
            }
        }
        d[i] = d[maxInd] + 1;
        p[i] = maxInd;
    }

    int maxInd = 0;
    for(int i = 1; i < n; ++i){
        if(d[i] > d[maxInd]){
            maxInd = i;
        }
    }
    
    vector<int> r = {a[maxInd]};
    int ind = p[maxInd];
    while(ind >= 0){
        r.push_back(a[ind]);
        ind = p[ind];
    }

    reverse(r.begin(), r.end());

    return r;
}

int main(){
    int n;
    cin >> n;

    vector<int> a(n);
    for(auto &el : a){
        cin >> el;
    }

    vector<int> r = lis(a);

    for(auto i : r){
        cout << i << ' ';
    }
    cout << endl;
}