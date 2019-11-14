#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <unordered_map>

using namespace std;

pair<vector<vector<int>>, vector<vector<int>>> cover(const vector<int> a){
    int n = a.size();

    vector<vector<int>> c;
    vector<vector<int>> p;

    for(int i = 0; i < n; ++i){
        int j = 0;
        while(j < c.size() && c[j].back() < a[i]){
            j++;
        }

        if(j == c.size()){
            c.push_back(vector<int>());
            p.push_back(vector<int>());
        }

        c[j].push_back(a[i]);

        if(j > 0){
            p[j].push_back(c[j - 1].size() - 1);
        }
        else{
            p[j].push_back(0);
        }
    }

    return make_pair(c, p);
}

vector<int> lis(const vector<int> a){
    vector<vector<int>> c;
    vector<vector<int>> p;

    tie(c, p) = cover(a);

    int cSize = c.size(); 

    vector<int> r;
    int j = c[cSize - 1].size() - 1;
    for(int i = cSize - 1; i >= 0; --i){
        r.push_back(c[i][j]);
        j = p[i][j];
    }

    reverse(r.begin(), r.end());

    return r;
}

string lcp(const string &s1, const string &s2){
    unordered_map<char, vector<int>> dict;

    for(auto symb : s1){
        dict[symb]; // creates vector I hope
    }

    for(int i = s2.size() - 1; i >= 0; --i){
        auto it = dict.find(s2[i]);
        if(it != dict.end()){
            it->second.push_back(i);
        }
    }

    for(auto &el : dict){
        cout << el.first << ": {";
        for(auto i : el.second){
            cout << i << ' ';
        }
        cout << "}" << endl;
    }

    vector<int> a;
    for(auto symb : s1){
        vector<int> ref = dict[symb];
        // a.insert(a.begin(), ref.begin(), ref.end()); // вставит перед a.begin()
        a.insert(a.end(), ref.begin(), ref.end()); // а надо перед a.end()
    }

    for(auto i : a){
        cout << i << ' ';
    }
    cout << endl;

    vector<int> r = lis(a);

    string res;
    for(auto i : r){
        cout << i << ' ';
        res.push_back(s2[i]);
    }
    cout << endl;

    return res;
}

int main(){

    string s1, s2;
    cin >> s1 >> s2;

    cout << lcp(s1, s2) << endl;
}
