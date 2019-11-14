#include <iostream>
#include <vector>
#include <limits>
#include <cassert>

using namespace std;

static const int inf = numeric_limits<int>::max();

struct edge{
    int v;
    int c;
    int f;
    edge(int vv, int cc, int ff) : v(vv), c(cc), f(ff) {}
};

int dfs(vector<vector<int>> &c, vector<vector<int>> &f, int u, int t, vector<int> &path, vector<int> &used) {
    used[u] = 1;
    path.push_back(u);

    if(u == t){
        return 1;
    }

    int n = c.size();
    for(int v = 0; v < n; ++v){
        if(!used[v] && c[u][v] > 0){
            if(dfs(c, f, v, t, path, used)){
                return 1;
            }
        }
    }
    
    path.pop_back();
    return 0;
}

int ford_fulkerson(const vector<vector<edge>> &adj, int s, int t){
    int n = adj.size();
    vector<vector<int>> c(n), f(n);
    for(int i = 0; i < n; ++i){
        c[i].assign(n, 0);
        f[i].assign(n, 0);
    }

    for(int u = 0; u < n; ++u){
        for(auto &e : adj[u]){
            c[u][e.v] = e.c;
        }
    }

    vector<int> used(n, 0);
    vector<int> path;
    while (dfs(c, f, s, t, path, used)) {
        used.assign(n, 0);

        assert(!path.empty());

        cout << "path:" << endl;
        for(int i : path){
            cout << i << ' ';
        }

        int size = path.size();

        int minC = c[path[0]][path[1]];
        for(int i = 1; i < size - 1; ++i){
            int u = path[i];
            int v = path[i + 1];
            if (c[u][v] < minC) {
                minC = c[u][v];
            }
        }

        cout << " | " << minC << endl;

        for(int i = 0; i < size - 1; ++i){
            int u = path[i];
            int v = path[i + 1];
            c[u][v] -= minC;
            c[v][u] += minC;
            f[u][v] += minC;
            f[v][u] -= minC;
        }

        path.clear();
    }

    int flow = 0;
    for(int i = 0; i < n; ++i){
        flow += f[i][t];
    }

    return flow;
}

int main(){
    int v, e;
    cin >> v >> e;

    vector<vector<edge>> adj(v);

    for(int i = 0; i < e; ++i){
        int u, v, c;
        cin >> u >> v >> c;

        edge tmp(v, c, 0);
        adj[u].push_back(tmp);
    }

    int s, f;
    cin >> s >> f;
    cout << "maximal flow from " << s << " to " << f << ": " << endl;

    cout << ford_fulkerson(adj, s, f) << endl;
}