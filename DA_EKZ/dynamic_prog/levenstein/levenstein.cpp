#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int levensteinDistance(const string &a, const string &b){
    int m = a.length();
    int n = b.length();

    vector<vector<int>> d(m + 1);
    for(auto &v : d){
        v = vector<int>(n + 1);
    }

    for(int i = 0; i <= m; ++i){
        d[i][0] = i;
    }

    for(int j = 1; j <= n; ++j){
        d[0][j] = j;
    }

    for(int i = 1; i <= m; ++i){
        for(int j = 1; j <= n; ++j){
            if(a[i - 1] == b[j - 1]){
                d[i][j] = d[i - 1][j - 1];
            }
            else{
                d[i][j] = min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1});
            }
        }
    }

    for(auto &v : d){
        for(auto el : v){
            cout << el << ' ';
        }
        cout << endl;
    }

    return d[m][n]; 
}

int main(){
    string a, b;
    cin >> a >> b;

    cout << a << endl;
    cout << b << endl;

    cout << levensteinDistance(a, b) << endl;
}
