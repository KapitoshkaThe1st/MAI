#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

long long solution(std::vector<std::vector<long long>> &mem, const std::string &str, int left, int right) {
    if(mem[left][right - left] != 0){
        return  mem[left][right - left];
    }

    long long inter = right - left == 1 ? 0 : solution(mem, str, left + 1, right - 1);
    mem[left][right - left] = solution(mem, str, left + 1, right) + solution(mem, str, left, right - 1) - inter;
    if(str[left] == str[right]){
        mem[left][right - left] += inter + 1;
    }
    return mem[left][right - left];
}

long long countOfUniqueSubPalindromes(const std::string &str) {
    int len = str.length();
    std::vector<std::vector<long long>> mem(len);

    for(int i = 0; i < len; i++){
        mem[i].resize(len - i, 0);
    }

    for(int i = 0; i < len; i++){
        mem[i][0] = 1;
    }

    return solution(mem, str, 0, len - 1);
}

int main() {
    std::string str;
    std::cin >> str;

    std::cout << countOfUniqueSubPalindromes(str) << std::endl;

    return 0;
}
