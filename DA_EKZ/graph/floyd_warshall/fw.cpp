#include <iostream>
#include <vector>
#include <limits>
#include <tuple>

using namespace std;

static const int inf = numeric_limits<int>::max();

typedef vector<vector<int>> mat;

void print_mat(const mat &m) {
    for(auto &row : m){
        for(auto el : row){
            if(el == inf){
                cout << "inf";
            }
            else{
                cout << el;
            }
            cout << '\t'; 
        }
        cout << endl;
    }
}

void aux_recursive(const vector<vector<int>> &p, int s, int f, vector<int> &res){
    int k = p[s][f];
    // cout << k << endl;
    if(k == -1){
        return;
    }
    aux_recursive(p, s, k, res);
    res.push_back(k);
    aux_recursive(p, k, f, res);
}

pair<int, vector<int>> floyd_warshall(const vector<vector<int>> &w, int s, int f){
    int n = w.size();

    vector<vector<int>> d = w, p(n);

    for(auto &v : p){
        v.assign(n, -1);
    }

    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (d[i][k] < inf && d[k][j] < inf) // при отрицательных ребрах спасает от появления значений inf - k, которые, очевидно, все равно бесконечны
                    if (d[i][k] + d[k][j] < d[i][j]){
                        d[i][j] = d[i][k] + d[k][j];
                        p[i][j] = k;
                    }

    vector<int> res;
    res.push_back(s);
    aux_recursive(p, s, f, res);
    res.push_back(f);

    return make_pair(d[s][f], res);
}

int main(){
    int v, e;
    cin >> v >> e;

    vector<vector<int>> w(v);
    for(auto &el : w){
        el.assign(v, inf);
    }

    for(int i = 0; i < v; ++i){
        w[i][i] = 0;
    }

    for(int i = 0; i < e; ++i){
        int u, v;
        cin >> u >> v >> w[u][v];
    }

    print_mat(w);
    cout << "---" << endl;

    int s, f;
    cin >> s >> f;
    cout << "from " << s << " to " << f << endl;

    int d;
    vector<int> path;

    tie(d, path) = floyd_warshall(w, s, f);

    cout << d << endl;
    for(int i : path){
        cout << i << ' ';
    }
    cout << endl;
}