#include <iostream>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

static const int inf = numeric_limits<int>::max();

struct dj_edge {
    int v;
    int w;
};

pair<vector<int>, vector<int>> dijkstra(vector<vector<dj_edge>> &adj, int n, int s) {
    static const int inf = numeric_limits<int>::max();
    vector<int> d(n, inf), p(n, -1), used(n);

    d[s] = 0;
    priority_queue<pair<int, int>> pq;
    pq.push(make_pair(0, s));

    while (!pq.empty()) {
        int curU, curD;
        tie(curD, curU) = pq.top();
        pq.pop();
        if (used[curU]) {
            continue;
        }

        for (dj_edge &e : adj[curU]) {
            if(used[e.v]){
                continue;
            }
            if (d[curU] + e.w < d[e.v]) {
                d[e.v] = d[curU] + e.w;
                p[e.v] = curU;
                pq.push(make_pair(-d[e.v], e.v));  // минус, чтобы не делать возрастающую (выше - меньше) кучу с помощью всяких std::greater()
            }
            used[curU] = 1;
        }
    }

    // cout << "dijkstra finished" << endl;
    return make_pair(d, p);
}

struct bf_edge {
    int u;
    int v;
    int w;
};

vector<int> bellman_ford(vector<bf_edge> &e, int n, int s) {
    static const int inf = numeric_limits<int>::max();
    vector<int> d(n, inf), p(n, -1);
    d[s] = 0;

    for (int i = 0; i < n - 1; ++i) {
        for (auto &el : e) {
            if (d[el.u] < inf && d[el.u] + el.w < d[el.v]) {
                d[el.v] = d[el.u] + el.w;
                p[el.v] = el.u;
            }
        }
    }

    for (auto &el : e) {
        if (d[el.u] < inf && d[el.u] + el.w < d[el.v]) {
            return vector<int>();
        }
    }

    // cout << "bf finished" << endl;
    return d;
}

pair<int, vector<int>> johnson(const vector<vector<dj_edge>> &adj, int s, int f){
    int n = adj.size();

    vector<bf_edge> edges;
    for(int i = 0; i < n; ++i){
        for(auto &e : adj[i]){
            bf_edge tmp;
            tmp.u = i;
            tmp.v = e.v;
            tmp.w = e.w;
            edges.push_back(tmp);
        }
    }

    int fictive = n;

    for(int i = 0; i < n; ++i){
        bf_edge tmp;
        tmp.u = fictive;
        tmp.v = i;
        tmp.w = 0;
        edges.push_back(tmp);
    }

    vector<int> h = bellman_ford(edges, n + 1, fictive);

    if (h.empty()){
        return {};
    }

    vector<vector<dj_edge>> newAdj = adj;
    for(int i = 0; i < n; ++i){
        for(auto &e : newAdj[i]){
            e.w = e.w + h[i] - h[e.v]; 
        }
    }

    vector<vector<int>> d(n), p(n);
    for(int i = 0; i < n; ++i){
        tie(d[i], p[i]) = dijkstra(newAdj, n, i);
    }

    for(int u = 0; u < n; ++u){
        for(int v = 0; v < n; ++v){
            d[u][v] = d[u][v] - h[u] + h[v];
        }
    }

    // cout << "here" << endl;
    vector<int> path;
    int k = f;
    while(k != -1){
        path.push_back(k);
        k = p[s][k];
    }
    // path.push_back(s);

    reverse(path.begin(), path.end());

    return make_pair(d[s][f], path);
}

int main() {
    int v, e;
    cin >> v >> e;

    vector<vector<dj_edge>> adj(v);

    for (int i = 0; i < e; ++i) {
        dj_edge edge;
        int u;
        cin >> u >> edge.v >> edge.w;
        adj[u].push_back(edge);
    }

    int s, f;
    cin >> s >> f;
    cout << "from " << s << " to " << f << endl;

    int d;
    vector<int> path;

    tie(d, path) = johnson(adj, s, f);

    cout << d << endl;
    for (int i : path) {
        cout << i << ' ';
    }
    cout << endl;
}
