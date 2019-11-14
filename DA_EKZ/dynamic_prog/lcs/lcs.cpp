#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>

using namespace std;

pair<int, vector<vector<int>>> lcs(const string &x, const string &y){
    int m = x.length();
    int n = y.length();
    
    vector<vector<int>> c(m + 1), b(m);
    for(auto &v : c){
        v = vector<int>(n + 1);
    }

    for(auto &v : b){
        v = vector<int>(n);
    }

    for(int i = 1; i <= m; ++i){
        c[i][0] = 0;
    }
    
    for(int j = 0; j <= n; ++j){
        c[0][j] = 0;
    }

    for(int i = 1; i <= m; ++i){
        for(int j = 1; j <= n; ++j){
            if(x[i] == y[j]){
                c[i][j] = c[i - 1][j - 1] + 1;
                b[i - 1][j - 1] = 1;
            }
            else{
                if(c[i - 1][j] > c[i][j - 1]){
                    c[i][j] = c[i - 1][j];
                    b[i - 1][j - 1] = 2;
                }
                else{
                    c[i][j] = c[i][j - 1];
                    b[i - 1][j - 1] = 0;
                }
            }
        }
    }
    return make_pair(c[m][n], b);
}

string restoreLcs(const string &x, const string &y, const vector<vector<int>> &b){
    int i = x.length() - 1;
    int j = y.length() - 1;

    string r;

    while(i >= 0 && j >= 0){
        if(b[i][j] == 1){
            r.push_back(x[i]);
            i--;
            j--;
        }
        else if(b[i][j] == 2){
            i--;
        }
        else{
            j--;
        }
    }

    reverse(r.begin(), r.end());
    return r;
}

int main(){
    string x,y;
    cin >> x >> y;

    cout << x << endl;
    cout << y << endl;

    int l;
    vector<vector<int>> b;

    tie(l, b) = lcs(x, y);

    for(auto &v : b){
        for(auto el : v){
            cout << el << ' ';
        }
        cout << endl;
    }

    cout << l << endl;
    cout << restoreLcs(x, y, b) << endl;
}