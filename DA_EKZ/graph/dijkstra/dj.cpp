#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <tuple>
#include <queue>
#include <cassert>

using namespace std;

struct edge{
    int v;
    int w;
};

pair<int, vector<int>> dijkstra(vector<vector<edge>> &adj, int n, int s, int f){
    static const int inf = numeric_limits<int>::max();
    vector<int> d(n, inf), p(n, -1), used(n);
    
    d[s] = 0;
    priority_queue<pair<int, int>> pq;
    pq.push(make_pair(0, s));

    while(!pq.empty()){
        int curU, curD;
        tie(curD, curU) = pq.top();
        pq.pop();
        if(used[curU]){
            continue;
        }

        cout << curU << endl;

        for(edge &e : adj[curU]){
            if(used[e.v]){       // если вершина использована, то до нее кратчайший путь уже найден, и этим мы обеспечиваем его сохранность при обработке следующих вершин и ребер (в том числе уже неактуальных в очереди с приоритетом)
                continue;
            }
            
            if((d[curU] + e.w < d[e.v])){
                d[e.v] = d[curU] + e.w;
                p[e.v] = curU;
                pq.push(make_pair(-d[e.v], e.v)); // минус, чтобы не делать возрастающую (выше - меньше) кучу с помощью всяких std::greater()
            }
            used[curU] = 1; 
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

    vector<vector<edge>> adj(n);

    for(int i = 0; i < m; ++i){
        int u;
        edge e;
        cin >> u >> e.v >> e.w;
        adj[u].push_back(e);
    }

    int s, f;
    cin >> s >> f;

    cout << "from " << s << " to " << f << endl;
    vector<int> res;
    int d;
    tie(d, res) = dijkstra(adj, n, s, f);

    cout << "----" << endl;

    cout << d << endl;
    for(int i : res){
        cout << i << ' ';
    }
    cout << endl;
}
