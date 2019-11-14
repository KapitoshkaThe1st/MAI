#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

void dfs(const vector<vector<int>> &adj, vector<int> &color, int s){
    color[s] = 1;
    int size = adj[s].size();
    for(int i = 0; i < size; ++i){
        int v = adj[s][i];
        if(color[v] == 0){
            dfs(adj, color, v);
        }
    }
    color[s] = 2;
}

void dfs_time(const vector<vector<int>> &adj, vector<int> &color, vector<int> &d, vector<int> &f, int &t, int u){
    color[u] = 1;
    d[u] = t;
    t++;
    int size = adj[u].size();
    for(int i = 0; i < size; ++i){
        int v = adj[u][i];
        if(color[v] == 0){
            dfs_time(adj, color, d, f, t, v);
        }
    }
    color[u] = 2;
    f[u] = t;
    t++;
}

int topological_aux_dfs(const vector<vector<int>> &adj, vector<int> &color, vector<int> &res, int u){
    color[u] = 1;
    int size = adj[u].size();
    int r = 1;
    for(int i = 0; i < size; ++i){
        int v = adj[u][i];
        if(color[v] == 1){
            return 0;
        }
        if(color[v] == 0){
            if(topological_aux_dfs(adj, color, res, v) == 0){
                return 0;
            }
        }
    }
    res.push_back(u);
    color[u] = 2;
    return 1;
}

vector<int> topological_sort(const vector<vector<int>> &adj){ // с проверкой наличия циклов, если есть цикл, то { }
    int n = adj.size();
    vector<int> color(n);
    vector<int> res;

    for(int i = 0; i < n; ++i){
        if(color[i] == 0){
            if(topological_aux_dfs(adj, color, res, i) == 0){
                return {};
            }
        }
    }

    reverse(res.begin(), res.end());
    return res;
}

// -- Strongly connected component -- 
 
vector<vector<int>> transpose(const vector<vector<int>> &adj){
    int n = adj.size();
    vector<vector<int>> res(n);

    for(int u = 0; u < n; ++u){
        int size = adj[u].size();
        for(int i = 0; i < size; ++i){
            int v = adj[u][i];
            res[v].push_back(u);
        }
    }

    return res;
}

void scc_aux_dfs1(const vector<vector<int>> &adj, vector<int> &used, int u, vector<int> &t){

    used[u] = 1;
    int size = adj[u].size();
    for(int i = 0; i < size; ++i){
        int v = adj[u][i];
        if(!used[v]){
            scc_aux_dfs1(adj, used, v, t);
        }
    }
    t.push_back(u);
}

void scc_aux_dfs2(const vector<vector<int>> &adj, vector<int> &used, int u, vector<int> &component){

    used[u] = 1;
    int size = adj[u].size();
    for(int i = 0; i < size; ++i){
        int v = adj[u][i];
        if(!used[v]){
            scc_aux_dfs2(adj, used, v, component);
        }
    }
    component.push_back(u);
}

vector<vector<int>> strongly_connected_components(const vector<vector<int>> &adj){
    int n = adj.size();

    // topological sort
    vector<int> t;
    vector<int> used(n, 0);

    for(int i = 0; i < n; ++i){
        if(!used[i]){
            vector<int> component;
            scc_aux_dfs1(adj, used, i, t);
        }
    }
    reverse(t.begin(), t.end());

    // transposed graph
    vector<vector<int>> adjTransposed = transpose(adj);

    // 2nd dfs series
    used.assign(n, 0);
    vector<vector<int>> res;
    for(int i = 0; i < n; ++i){
        if(!used[t[i]]){
            vector<int> component;
            scc_aux_dfs2(adjTransposed, used, t[i], component);
            res.push_back(component);
        }
    }

    return res;
} 
// -- -- 
int main(){
    int v, e;
    cin >> v >> e;

    vector<vector<int>> adj(v);
    for(int i = 0; i < e; ++i){
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    int s;
    cin >> s;

    vector<int> c(v), d(v), f(v);
    int t = 0;

    for(int i = 0; i < v; ++i){
        if(c[i] == 0){
            dfs_time(adj, c, d, f, t, i);
        }
    }

    cout << "discovery / leave time: " << endl;
    for(int i = 0; i < v; ++i){
        cout << i << ": " << d[i] << " / " << f[i] << endl; 
    }

    vector<int> res = topological_sort(adj);

    cout << "topological sort: " << endl;
    cout << "{ ";
    for(int i : res){
        cout << i << ' ';
    }
    cout << "}" << endl;

    vector<vector<int>> sccs = strongly_connected_components(adj);

    cout << "strongly connected components: " << endl;
    cout << "{";
    for(auto &vec : sccs){
        cout << " { ";
        for(int i : vec){
            cout << i << ' '; 
        }
        cout << "}";
    }
    cout << " }" << endl;

}