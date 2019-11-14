#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <tuple>

using namespace std;

pair<string, int> burrows_wheeler_naive(const string &str){
    int n = str.length();
    vector<string> s;
    s.reserve(n);

    s.push_back(str);
    for(int i = 1; i < n; ++i){
        string &ref = s.back(); 
        s.push_back(ref.back() + ref.substr(0, n - 1));
    }

    sort(s.begin(), s.end());

    string res;
    for(auto &el : s){
        res.push_back(el.back());
    }

    int k = 0;
    while(s[k] != str){
        k++;
    }

    return make_pair(res, k);
}

string reverse_burrows_wheeler(const string &str, int k){
    string sorted = str;
    sort(sorted.begin(), sorted.end());

    int n = str.length();
    string res;
    res.reserve(n);
    int i = k;
    do{
        char c = sorted[i];
        int count = 0;
        for(int j = 0; j <= i; ++j){
            if(sorted[j] == c){
                count++;
            }
        }

        int j = 0;
        while(count > 0){
            if(str[j] == c){
                count--;
            }
            j++;
        }
        i = j - 1;
        res.push_back(c);

    }while(i != k);

    return res;
}

int main(){
    string str;
    cin >> str;

    string res;
    int k;
    tie(res, k) = burrows_wheeler_naive(str);

    cout << "BWT: " << endl;
    cout << k << endl;
    cout << res << endl;

    cout << "reverse BWT: " << endl;
    cout << reverse_burrows_wheeler(res, k) << endl;
}
