#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

using ulong = unsigned long;
const ulong INF = std::numeric_limits<ulong>::max();

struct TDist {
    int to;
    ulong len;

    TDist(int toParam, ulong lenParam) : to(toParam), len(lenParam) {}
    friend bool operator>(const TDist &d1, const TDist &d2) {
        return d1.len > d2.len;
    }
};

using TGraph = std::vector<std::vector<TDist>>;

std::vector<ulong> Dijkstra(const TGraph &graph, int start) {
    int vertCount = graph.size();
    std::vector<ulong> dist(vertCount, INF);
    dist[start] = 0;

    std::vector<bool> used(vertCount, false);

    std::priority_queue<TDist, std::vector<TDist>, std::greater<TDist>> prior;
    prior.push(TDist(start, 0));

    while(!prior.empty()){
        int minDistVert = prior.top().to;
        prior.pop();

        if(used[minDistVert]){
            continue;
        }

        used[minDistVert] = true;
        ulong minDist = dist[minDistVert];

        if(minDistVert == INF){
            break;
        }

        int edgeCount = graph[minDistVert].size();
        for(int i = 0; i < edgeCount; i++){
            int to = graph[minDistVert][i].to;
            ulong newDist = graph[minDistVert][i].len + minDist;

            if(dist[to] > newDist){
                dist[to] = newDist;
                prior.push(TDist(to, newDist));
            }
        }

    }
    return dist;
}

int main() {
    int m, n, start, finish;
    std::cin >> n >> m >> start >> finish;

    start--;
    finish--;

    TGraph adj(n);
    for (int i = 0; i < m; i++) {
        int from, to, len;
        std::cin >> from >> to >> len;

        from--;
        to--;

        adj[from].push_back(TDist(to, len));
        adj[to].push_back(TDist(from, len));
    }

    std::vector<ulong> dist = Dijkstra(adj, start);
    if (dist[finish] != INF) {
        std::cout << dist[finish];
    }
    else {
        std::cout << "No solution";
    }
    std::cout << std::endl;
}
