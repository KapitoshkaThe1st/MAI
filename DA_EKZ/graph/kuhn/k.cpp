#include <iostream>
#include <vector>

using namespace std;

int dfs(const vector<vector<int>> &adj, vector<int> &match, vector<int> &used, int u){
    used[u] = 1;
    for(int v : adj[u]){
        if(match[v] == -1 || (!used[match[v]] && dfs(adj, match, used, match[v]))){
            match[v] = u;
            return 1;
        }
    }
    return 0;
}

vector<int> kuhn(const vector<vector<int>> &adj){
    int n = adj.size();
    vector<int> match(n, -1), used;

    for(int i = 0; i < n; ++i){
        used.assign(n, 0);
        dfs(adj, match, used, i);
    }

    return match;
}

int main(){
    int v, e;
    cin >> v >> e;

    vector<vector<int>> adj(v);

    for(int i = 0; i < e; ++i){
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v); 
    }

    vector<int> m = kuhn(adj);

    vector<pair<int, int>> edges;
    for(int i = 0; i < v; ++i){
        // cout << m[i] << ' ';
        if(m[i] != -1){
            edges.push_back(make_pair(i, m[i]));
        }
    }
    // cout << endl;

    for(auto &el : edges){
        cout << el.first << " - " << el.second << endl;
    }
}