#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

using namespace std;

pair<int, vector<float>> knapsack(int n, const vector<int> &w, const vector<int> &p, int limW) {
    vector<pair<float, int>> c(n);

    for(int i = 0; i < n; ++i){
        c[i] = make_pair((float)p[i] / w[i], i);
    }

    // sort(c.begin(), c.end(), ::greater<pair<float, int>>()); // делают одно и то же
    sort(c.rbegin(), c.rend());

    // for(auto &el : c){
    //     cout << el.second << " - " << el.first << endl;
    // }

    int i = 0;
    float s = 0.0f;
    vector<float> r(n);

    while(limW > 0 && i < n){
        int ind = c[i].second;
        if(w[ind] < limW){
            s += p[ind];
            limW -= w[ind];
            r[ind] = w[ind];
            i++;
        }
        else{
            s += limW * c[i].first;
            r[ind] = limW;
            limW = 0;
        }
    }

    return make_pair(s, r);
}

int main() {
    int n;

    cin >> n;
    vector<int> w(n), p(n);

    for (auto &el : w) {
        cin >> el;
    }

    for (auto &el : p) {
        cin >> el;
    }

    int limW;
    cin >> limW;

    float s;
    vector<float> r;
    tie(s, r) = knapsack(n, w, p, limW);

    cout << s << endl;
    for (auto el : r) {
        cout << el << ' ';
    }
    cout << endl;
}