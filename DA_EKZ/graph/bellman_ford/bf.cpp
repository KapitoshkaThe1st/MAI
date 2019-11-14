#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <tuple>

using namespace std;

struct edge{
    int u;
    int v;
    int w;
};

pair<int, vector<int>> bellman_ford(vector<edge> &e, int n, int s, int f){
    static const int inf = numeric_limits<int>::max();
    vector<int> d(n, inf), p(n, -1);
    d[s] = 0;

    for(int i = 0; i < n - 1; ++i){
        for(auto &el : e){
            if(d[el.u] < inf && d[el.u] + el.w < d[el.v]){
                d[el.v] = d[el.u] + el.w;
                p[el.v] = el.u;
            }
        }
    }

    for(auto &el : e){
        if(d[el.u] < inf && d[el.u] + el.w < d[el.v]){
            return make_pair(-1, vector<int>());
        }
    }

    vector<int> r;
    int k = f;
    while(k != -1){
        r.push_back(k);
        k = p[k];
    }

    reverse(r.begin(), r.end());

    return make_pair(d[f], r);
}

int main(){
    int n, m;
    cin >> n >> m;

    vector<edge> e(m);
    for(auto &el : e){
        cin >> el.u >> el.v >> el.w;
    }

    int s, f;
    cin >> s >> f;

    vector<int> res;
    int d;
    tie(d, res) = bellman_ford(e, n, s, f);

    if(d != -1){
        cout << d << endl;
        for(int i : res){
            cout << i << ' ';
        }
        cout << endl;
    }
    else{
        cout << "there is a negative cycle i think..." << endl;
    }
    
}
