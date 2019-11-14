#include <iostream>
#include <vector>
#include <string>

using namespace std;

int aux(int n, vector<int> &c, const string &s) {

    if(n == 0){
        return 0;
    }
    if (n < 0 || s[n] == 'W') {
        return -1;
    }
    c[n] = max(max(aux(n - 1, c, s), aux(n - 3, c, s)), aux(n - 5, c, s));
    if(c[n] == -1){
        return -1;
    }
    if(s[n] == '\"'){
        c[n]++;
    }
    return c[n];
}

int solution(const string &s) {
    int n = s.length();
    vector<int> c(n, -1);
    return aux(n - 1, c, s);
}

int main(){
    string s;
    cin >> s;
    cout << solution(s) << endl;
    return 0;
}