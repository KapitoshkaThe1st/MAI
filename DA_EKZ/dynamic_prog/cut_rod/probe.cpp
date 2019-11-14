#include <iostream>
#include <vector>

using namespace std;

int func(vector<int> &v){
    for(int i = 0; i < 10; ++i){
        v.push_back(i);
    }
    return 1488;
}

int main(){
    vector<int> vec;

    auto p = make_pair(func(vec), vec);

    cout << p.first << endl;
    for(auto num : p.second){
        cout << num << ' ';
    }
    cout << endl;
}