#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

vector<int> lis(const vector<int> a){
    int n = a.size();

    vector<int> d(n, numeric_limits<int>::max()), l(n);

    int m = 0;
    for(int i = 0; i < n; ++i){ // n раз
        int k = 0;
        while(d[k] < a[i]){ // O(n) линейный поиск 
            k++;
        }
        if(k > m){
            m++;
        }
        d[k] = a[i];
        l[i] = k;
    }
    // всего O(n^2)

    for(auto i : l){
        cout << i << ' ';
    }
    cout << endl;

    int maxInd = 0;
    for(int i = 1; i < n; ++i){
        if(l[i] > l[maxInd]){
            maxInd = i;
        }
    }

    cout << maxInd << endl;

    vector<int> r;
    int i = maxInd;
    while(l[i] > 0){
        r.push_back(a[i]);
        for(int j = i - 1; j >= 0; --j){
            if((l[j] == l[i] - 1) && a[j] < a[i]){
                i = j;
                break;
            }
        }
    }
    r.push_back(a[i]);


    reverse(r.begin(), r.end());
    return r;
}

// здесь где-то ошибка (вероятнее всего в бинарном поиске), а так же присутствует ошиб очка в развитии
// ошибка где-то ниже

int binSearchCeil(const vector<int> &a, int s, int f, int k){
    // cout << "-----" << endl;
    // cout << "s m f" << endl;
    while(s <= f){
        int m = (s + f) / 2;
        // cout << s << ' ' << m << ' ' << f << endl;
        if(k >= a[m]){
            s = m + 1;
            // cout << '<' << endl;
        }
        else{
            f = m - 1;
            // cout << ">=" << endl;
        }
    }
    // cout << "-----" << endl;
    // cout << endl;
    // cout << s << ' ' << f << endl;
    return s;
}

vector<int> fast_lis(const vector<int> a){
    int n = a.size();

    vector<int> d(n, numeric_limits<int>::max()), l(n);

    int m = 0;
    for(int i = 0; i < n; ++i){ // n раз
        int k = binSearchCeil(a, 0, m, a[i]);
        if(k > m){
            m++;
        }
        d[k] = a[i];
        l[i] = k;
    }
    // всего O(n^2)

    for(auto i : l){
        cout << i << ' ';
    }
    cout << endl;

    int maxInd = 0;
    for(int i = 1; i < n; ++i){
        if(l[i] > l[maxInd]){
            maxInd = i;
        }
    }

    cout << maxInd << endl;

    vector<int> r;
    int i = maxInd;
    while(l[i] > 0){
        r.push_back(a[i]);
        for(int j = i - 1; j >= 0; --j){
            if((l[j] == l[i] - 1) && a[j] < a[i]){
                i = j;
                break;
            }
        }
    }
    r.push_back(a[i]);


    reverse(r.begin(), r.end());
    return r;
}

// ошибка где-то выше

int lisLength(const vector<int> a){
    int n = a.size();

    vector<int> d(n, numeric_limits<int>::max()), p(n);

    for(int i = 0; i < n; ++i){
        int k = (int)(upper_bound(d.begin(), d.end(), a[i], std::less_equal<int>()) - d.begin());
        d[k] = a[i];
    }

    for(auto i : d){
        cout << i << ' ';
    }
    cout << endl;
    
    return (int)(lower_bound(d.begin(), d.end(), numeric_limits<int>::max()) - d.begin());
}

int main(){
    // //                 0  1  2  3  4  5   6   7   8 
    // vector<int> arr = {1, 4, 8, 8, 9, 13, 14, 56, 72, 1488};

    // for(auto i : arr){
    //     cout << '\t' << i << ' ' << binSearchCeil(arr, 0, arr.size() - 1, i) << endl;
    // }

    // return 0;

    int n;
    cin >> n;

    vector<int> a(n);
    for(auto &el : a){
        cin >> el;
        cout << el << ' ';
    }
    cout << endl;

    // cout << lisLength(a) << endl;

    vector<int> r = lis(a);

    for(auto i : r){
        cout << i << ' ';
    }
    cout << endl;

    r = fast_lis(a);

    for(auto i : r){
        cout << i << ' ';
    }
    cout << endl;

}