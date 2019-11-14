#include <iostream>
#include <vector>

using namespace std;

vector<int> move_to_front(const string &s){
    char a[26];
    for(int i = 0; i < 26; ++i){
        a[i] = i + 'a';
    }

    for(char c : a){
        cout << c << ' ';
    }
    cout << endl;

    int n = s.size();
    vector<int> res;
    for(int i = 0; i < n; ++i){
        int k = 0;
        while(a[k] != s[i]){
            k++;
        }
        res.push_back(k);
        for(int j = k; j > 0; --j){
            a[j] = a[j - 1];
        }
        a[0] = s[i];
    }
    return res;
}

int main(){
    string s;
    cin >> s;

    vector<int> res = move_to_front(s);

    for(int i : res){
        cout << i << ' ';
    }
    cout << endl;
}
