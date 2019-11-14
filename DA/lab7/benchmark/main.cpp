#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>

#include <random>
#include <ctime>

using namespace std;

int solution(std::vector<std::vector<int>> &mem, const std::string &str, int left, int right) {
    if (mem[left][right - left] != 0) {
        return mem[left][right - left];
    }

    int inter = right - left == 1 ? 0 : solution(mem, str, left + 1, right - 1);
    mem[left][right - left] = solution(mem, str, left + 1, right) + solution(mem, str, left, right - 1) - inter;
    if (str[left] == str[right]) {
        mem[left][right - left] += inter + 1;
    }
    return mem[left][right - left];
}

int countOfUniqueSubPalindromes(const std::string &str) {
    int len = str.length();
    std::vector<std::vector<int>> mem(len);

    for (int i = 0; i < len; i++) {
        mem[i] = std::vector<int>(len - i, 0);
    }

    for (int i = 0; i < len; i++) {
        mem[i][0] = 1;
    }

    return solution(mem, str, 0, len - 1);
}

void p(vector<int> v, vector<vector<int>> &s, int b, int e) {
    for (int i = b; i < e; i++) {
        if (i > b) {
            v.pop_back();
        }
        v.push_back(i);
        s.push_back(v);
        p(v, s, i + 1, e);
    }
}

int countOfUniqueSubPalindromesNaive(const std::string &str) {
    int len = str.length();

    vector<vector<int>> sets;
    p({}, sets, 0, str.length());

    int count = 0;

    for (auto &vec : sets) {
        int size = vec.size();
        int isPalindrom = true;
        for (int i = 0; i < size / 2; i++) {
            if (str[vec[i]] != str[vec[size - i - 1]]) {
                isPalindrom = false;
                break;
            }
        }
        if (isPalindrom) {
            count++;
        }
    }
    return count;
}

string RandString(vector<char> symbols, int len) {
    string res;
    res.reserve(len);

    int vecSize = symbols.size();

    for (int i = 0; i < len; i++) {
        int ind = rand() % vecSize;
        res += symbols[ind];
    }

    return res;
}

int main(int argc, char **argv) {
    srand(time(0));
    rand();
    // std::string str(argv[1]);
    // std::cin >> str;

    int lim = atoi(argv[1]);
    int step = atoi(argv[2]);

    vector<int> timeDyn;
    vector<int> timeNaive;

    for(int i = 5; i < lim; i += step){
        // int sum = 0;
        // for(int k = 0; k < 10; k++){
            string str = RandString({'a', 'b', 'o'}, i);

            chrono::time_point<chrono::system_clock> start, finish;

            start = chrono::system_clock::now();
            countOfUniqueSubPalindromes(str);
            finish = chrono::system_clock::now();

            // sum += chrono::duration_cast<chrono::microseconds>(finish - start).count();
        // }
        // timeDyn.push_back(sum / 10);
        
        timeDyn.push_back(chrono::duration_cast<chrono::microseconds>(finish - start).count());


        start = chrono::system_clock::now();
        countOfUniqueSubPalindromesNaive(str);
        finish = chrono::system_clock::now();

        timeNaive.push_back(chrono::duration_cast<chrono::microseconds>(finish - start).count());
    }

    int k = 5;
    for(auto el : timeDyn){
        cout << '(' << k << ';' << el << ')'; 
        k += step;
    }

    cout << "\n--------------------------" << endl;

    for (int i = 0; i < lim - 5; i++) {
        cout << '(' << i + 5 << ';' << timeNaive[i] << ')';
    }

    return 0;
}
