#include <iostream>
#include <vector>
#include <limits>
#include <tuple>

using namespace std;

typedef vector<vector<int>> mat;

static const int inf = numeric_limits<int>::max();

mat mult(const mat &a, const mat &b){
    int n = a.size();
    mat res(n);
    for(auto &el : res){
        el.assign(n, 0);
    }

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            for(int k = 0; k < n; ++k){
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

void print_mat(const mat &m){
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

mat extend_shortest_path(mat &r, const mat &m, mat &p){
    int n = m.size();
    mat res(n);
    for(auto &el : res){
        el.assign(n, 0);
    }
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            if(i == j){
                continue;
            }
            int minInd = 0;
            int minVal;
            if(r[i][minInd] == inf || m[minInd][j] == inf){
                minVal = inf;
            }
            else{
                minVal = r[i][minInd] + m[minInd][j];
            }

            for(int k = 1; k < n; ++k){
                int curVal;
                if(r[i][k] == inf || m[k][j] == inf){
                    curVal = inf;
                }
                else{
                    curVal = r[i][k] + m[k][j];
                }

                if(curVal < minVal){
                    minVal = curVal;
                    minInd = k;
                }
            }
            res[i][j] = minVal;
            if(minVal < r[i][j]){
                p[i][j] = minInd;
            }
        }
    }

    return res;
}

pair<mat, mat> shortest_path(const mat &m){
    int n = m.size();
    mat r = m; // m = l_1
    mat p(n);
    for(auto &v : p){
        v.assign(n, -1);
    }
    // инициализация предков для уже существующих ребер (т.е. для путей длины 1 и меньше) т.к. оно само ниоткуда не возьмется
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            if(i != j && m[i][j] < inf){
                p[i][j] = i; 
            }
        }
    }

    for(int i = 0; i < n - 2; ++i){
        r = extend_shortest_path(r, m, p);
    }

    return make_pair(r, p);
}

int main(){
    int v, e;
    cin >> v >> e;
    mat m(v);

    for(auto &el : m){
        el = vector<int>(v, inf);
    }

    for(int i = 0; i < v; ++i){
        m[i][i] = 0;
    }

    for(int i = 0; i < e; ++i){
        int u, v;
        cin >> u >> v >> m[u][v];
    }

    print_mat(m);

    int s, f;
    cin >> s >> f;
    
    mat r, p;

    tie(r, p) = shortest_path(m);

    cout << "Distances: " << endl;
    print_mat(r);

    cout << "shortest path from " << s << " to " << f << ": " << r[s][f] << endl;
    
    cout << "Predecessors: " << endl;
    print_mat(p);

    vector<int> path;
    int k = f;
    while(k != -1){
        path.push_back(k);
        k = p[s][k];
    }
    cout << "Shortest path: " << endl;
    for(auto it = path.rbegin(); it != path.rend(); ++it){
        cout << *it << ' ';
    }
    cout << endl;

}