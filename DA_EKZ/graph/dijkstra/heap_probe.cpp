#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <tuple>
#include <queue>
#include <cassert>

using namespace std;

int main(){
    struct s{
        int d;
        int v;
    };

    struct comp{
        bool operator()(const s &a, const s &b){
            return a.d > b.d;
        }
    };

    int n = 10;
    vector<s> ss(n, (s){1000, 0});

    for(int i = 0; i < n - 5; ++i){
        s tmp;
        tmp.d = n - i - 1;
        tmp.v = i;
        ss[i] = tmp;
    }

    make_heap(ss.begin(), ss.end(), comp());

    for(s el : ss){
        cout << "{ " << el.d << ", " << el.v << "} ";
    }
    cout << endl;

    ss.push_back((s){5, 5});
    push_heap(ss.begin(), ss.end(), comp());
    
    for(s el : ss){
        cout << "{ " << el.d << ", " << el.v << "} ";
    }
    cout << endl;

    // priority_queue<s, vector<s>, comp> pq(comp(), ss); 
}