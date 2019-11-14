#include <iostream>
#include <vector>

std::vector<int> TopologSort(std::vector<std::vector<int>> &graph) {
    int size = graph.size();
    std::vector<int> res;
    std::vector<int> indegree(size); // количество ребер для каждой вершины
    std::vector<int> zeroIndegree; // вершины без входящих ребер

    // подсчитываем количество входящих ребер для каждой вершины
    for(auto vec : graph){
        for(auto el : vec){
            indegree[el]++;
        }
    }

    // инициализируем стек вершин без входящих ребер
    for(int i = 0; i < size; i++){
        if(indegree[i] == 0){
            zeroIndegree.push_back(i);
        }
    }

    // пока есть вершины без входящих ребер
    // (если их сразу не оказалось, то нельзя 
    // отсортировать. Вернется пустой вектор)
    while(!zeroIndegree.empty()){
        // берем одну из них
        int vert = zeroIndegree.back();
        zeroIndegree.pop_back();

        // уменьшаем количество входящих ребер для каждой инцидентной вершины
        // т.е. как будто удаляем вершину и ребра из графа
        for(auto el : graph[vert]){
            indegree[el]--;
            // если у вершины не осталось входящих ребер, добавляем в список
            if(indegree[el] == 0){
                zeroIndegree.push_back(el);
            }
        }
        indegree[vert] = -1; // чтобы на следующих этапах не учитывалась
        res.push_back(vert); // добавляем в результат
    }

    // проверяем все ли вершины были обработаны
    // если не все, это значит в какой-то момент оказалось, что нельзя выбрать вершину без входящих ребер
    // то есть оставшиеся вершины составляют цикл, поэтому нельзя отсортировать
    for(auto el : indegree){
        if(el != -1){
            return {};
        }
    }

    return res;
}

int main() {
    int n, m;
    std::cin >> n;
    std::cin >> m;

    std::vector<std::vector<int>> graph(n);
    for (int i = 0; i < m; i++) {
        int from, to;
        std::cin >> from;
        std::cin >> to;
        graph[from - 1].push_back(to - 1);
    }

    std::vector<int> ans = TopologSort(graph);

    if (!ans.empty()) {
        for (auto el : ans) {
            std::cout << el + 1 << ' ';
        }
    }
    else {
        std::cout << -1;
    }
    std::cout << std::endl;

    return 0;
}
