#include <iostream>
#include <vector>
#include <string>

using namespace std;

int aux(int i, int j, vector<vector<int>> & c) {
    if(c[i][j] != -1){
        return c[i][j];
    }
    c[i][j] = 0;
    if(i - 1 >= 0 && j - 2 >= 0){
        c[i][j] += aux(i - 1, j - 2, c);
    }
    if (i - 2 >= 0 && j - 1 >= 0) {
        c[i][j] += aux(i - 2, j - 1, c);
    }

    return c[i][j];
}

int solution(int n, int m) {
    vector<vector<int>> c(n);
    for (int i = 0; i < n; ++i){
        c[i] = vector<int>(m, -1);
    }
    c[0][0] = 1;
    return aux(n - 1, m - 1, c);
}

int main(){
    int n, m;
    cin >> n >> m;
    cout << solution(n, m) << endl;
    return 0;
}