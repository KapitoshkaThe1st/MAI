#include <iostream>
#include <vector>

using namespace std;

int aux(int n, vector<int> &c, const vector<int> &p) {

    if(n <= 0){
        return 0;
    }

    if(c[n] != -1){
        return c[n];
    }

    c[n] = max(aux(n - 1, c, p), aux(n - 2, c, p)) + p[n - 1];
    return c[n];
}

int solution(int n, const vector<int> &p){
    vector<int> c(n + 1, -1);

    return aux(n, c, p);
}

int main(){
    int n;
    cin >> n;
    vector<int> p(n);
    for(int i = 0; i < n; ++i){
        cin >> p[i];
    }

    cout << solution(n, p) << endl;

    return 0;
}