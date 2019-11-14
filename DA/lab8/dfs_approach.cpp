#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int DFS(vector<char> &color, vector<vector<int>> &g, vector<int> &acum, int vert) {
    if(color[vert] == 'g'){
        return 1; // есть цикл
    }
    if(color[vert] == 'b'){
        return 0; // вершина уже просмотрена
    }
    color[vert] = 'g';
    for(auto el : g[vert]){
        if(DFS(color, g, acum, el)){
            return 1;
        }
    }
    color[vert] = 'b';
    acum.push_back(vert);
    return 0;
}

vector<int> TopologSort(vector<vector<int>> &g) {
    vector<int> res;
    int size = g.size();
    
    vector<char> color(size, 'w');

    for(int i = 0; i < size; i++){
        if(DFS(color, g, res, i)){
            return {};
        }
    }
    reverse(res.begin(), res.end());
    return res;
}

int main() {
    int n, m;
    cin >> n; 
    cin >> m;
    
    vector<vector<int>> graph(n);
    for(int i = 0; i < m; i++){
        int from, to;
        cin >> from;
        cin >> to;
        graph[from - 1].push_back(to - 1);
    }

    // for(int i = 0; i < n; i++){
    //     cout << i + 1 << " : ";
    //     for(auto el : graph[i]){
    //         cout << el + 1 << ' ';
    //     }
    //     cout << endl;
    // }

    vector<int> ans = TopologSort(graph);

    if(!ans.empty()){
        for(auto el : ans){
            cout << el + 1 << ' ';
        }
    }
    else{
        cout << -1;
    }
    cout << endl;

    return 0;
}