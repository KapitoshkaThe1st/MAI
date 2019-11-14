#include <iostream>
#include <vector>
#include <tuple>
#include <limits>

using namespace std;

pair<int, vector<vector<int>>>
BottomUpSolution(const vector<int> p, int n){
    vector<vector<int>> m(n);
    vector<vector<int>> s(n);

    for(auto &v : m){
        v = vector<int>(n, numeric_limits<int>::max());
    }

    for(auto &v : s){
        v = vector<int>(n);
    }

    for(int i = 0; i < n; i++){
        m[i][i] = 0;
    }

    for(int l = 2; l < n + 1; l++){
        for(int i = 0; i < n - l + 1; i++){
            int j = i + l - 1;
            for(int k = i; k < j; k++){
                int tmp = m[i][k] + m[k + 1][j] + p[i]*p[k + 1]*p[j + 1];
                if(tmp < m[i][j]){
                    m[i][j] = tmp;
                    s[i][j] = k;
                }
            }
        }
    }

    for(auto &v : m){
        for(auto el : v){
            cout << el << ' ';
        }
        cout << endl;
    }

    return make_pair(m[0][n - 1], s);
}

int main(){
    vector<int> p = {30, 35, 15, 5, 10, 20, 25};

    vector<vector<int>> s;
    int r;
    tie(r, s) = BottomUpSolution(p, 6);


    cout << "minimal operations count: " << r << endl;

}