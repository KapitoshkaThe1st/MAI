#include <iostream>
#include <vector>
#include <tuple>

using namespace std;

pair<int, vector<int>> knapsack(int n, const vector<int> &w, const vector<int> &p, int limW){
    vector<vector<int>> c(n + 1);

    for(auto &v : c){
        v = vector<int>(limW + 1);
    }

    for(int i = 0; i <= n; ++i){
        c[i][0] = 0;
    }

    for(int j = 1; j <= limW; ++j){
        c[0][j] = 0;
    }

    for(int i = 1; i <= n; ++i){
        for(int j = 1; j <= limW; ++j){
            if(j - w[i - 1] >= 0 && c[i - 1][j - w[i - 1]] + p[i - 1] > c[i - 1][j]){
                c[i][j] = c[i - 1][j - w[i - 1]] + p[i - 1];
            }
            else{
                c[i][j] = c[i - 1][j];
            }
        }
    }


    // for(auto &v : c){
    //     for(auto el : v){
    //         cout << el << ' ';
    //     }
    //     cout << endl;
    // }

    // restorting solution

    vector<int> s(n);
    int j = limW;    
    for(int i = n; i >= 1; --i){
        if(c[i][j] == c[i - 1][j]){
            s[i] = 0;
        }
        else{
            s[i] = p[i - 1]; // price of the i-th object means that we choose it (s[i] != 0)
            j -= w[i - 1];
        }
    }

    return make_pair(c[n][limW], s);
} 

int main(){
    int n;

    cin >> n;
    vector<int> w(n), p(n);

    for(auto &el : w){
        cin >> el;
    }

    for(auto &el : p){
        cin >> el;
    }

    int limW;
    cin >> limW;

    int c;
    vector<int> s;
    tie(c, s) = knapsack(n, w, p, limW);

    cout << c << endl;
    for(auto el : s){
        cout << el << ' ';
    }
    cout << endl;
}