#include <iostream>
#include <vector>
#include <tuple>

using namespace std;

int aux(int len, const vector<int> &prices, vector<int> &storage, vector<int> &parts){
    if(storage[len] != -1){
        return storage[len];
    }
    
    if(len == 0){
        return 0;
    }

    int max = -1;
    int maxLen = 0; 
    for(int i = 1; (i <= len && i < prices.size()); ++i){
        int tmp = prices[i] + aux(len - i, prices, storage, parts); 
        if(max < tmp){
            max = tmp;
            maxLen = i;
        }
    }
    parts[len] = maxLen;

    storage[len] = max;
    return max;
}

pair<int, vector<int>> solve(int len, const vector<int> &prices){
    vector<int> storage(len + 1, -1);
    vector<int> parts(len + 1);
    
    int res = aux(len, prices, storage, parts);
    
    vector<int> cuts;

    while(len > 0){
        cuts.push_back(parts[len]);
        len -= parts[len];
    }

    return make_pair(res, cuts); // !!!
}

int main(){
    int n;
    cin >> n;
    n++;

    vector<int> prices(n);
    for(int i = 0; i < n; ++i){
        cin >> prices[i];
    }


    int len;
    cin >> len;
    
    int res;
    vector<int> parts;

    tie(res, parts) = solve(len, prices);
    
    cout << res << endl;
    for(auto part : parts){
        cout << part << ' ';
    }
    cout << endl;

    return 0;
}
