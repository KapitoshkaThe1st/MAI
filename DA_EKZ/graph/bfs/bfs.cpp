#include <iostream>
#include <vector>
#include <queue>

using namespace std;

int bfs_d(const vector<vector<int>> &adj, int s, int f){
    int v = adj.size();

    vector<int> used(v), d(v, -1);
    queue<int> q;

    q.push(s);
    used[s] = 1;
    d[s] = 0;
    while(!q.empty()){
        int u = q.front();
        q.pop();
        int size = adj[u].size();
        for(int i = 0; i < size; ++i){
            int v = adj[u][i];
            if(!used[v]){
                q.push(v);
                d[v] = d[u] + 1;
                used[v] = 1;
            }
        }
    }

    return d[f];
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

    int s, f;
    cin >> s >> f;

    int d = bfs_d(adj, s, f);

    cout << d << endl;
}