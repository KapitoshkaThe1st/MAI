#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>

void ZPreprocess(std::vector<std::string> &str, std::vector<int> &z, int start);
void StringToUpper(std::string &str);

int main(){
    const std::string SENTINEL = "$";
    std::string line;
    std::string temp;
    std::vector<std::string> sum;

    getline(std::cin >> std::ws, line);
    StringToUpper(line);
    std::stringstream ss(line);

    int patternLength = 0;

    ss >> std::ws;
    while(!ss.eof()){
        ss >> temp >> std::ws;
        sum.push_back(temp);
        patternLength++;
    }
    sum.push_back(SENTINEL);

    std::vector<int> zFunc(3 * patternLength + 1);
    ZPreprocess(sum, zFunc, 1);

    int inPool = 0;
    int lineNumber = 0;
    int wordNumber = 0;
    int start = patternLength + 1;

    std::vector<std::pair<int, int>> wordCoord;

    while (1){
        if(ss.eof()){
            if (!getline(std::cin, line))
                break;

            wordNumber = 1;
            lineNumber++;
            if (line == "")
                continue;
            StringToUpper(line);
            ss.str(line);
            ss.clear();
        }

        ss >> std::ws;
        while (inPool < patternLength * 2 && !ss.eof()){
            ss >> temp >> std::ws;
            sum.push_back(std::move(temp));

            std::pair<int, int> tempPair(lineNumber, wordNumber);

            wordCoord.push_back(tempPair);
            
            wordNumber++;
            inPool++;
        }

        if(!ss.eof()){
            ZPreprocess(sum, zFunc, patternLength + 1);
            for(int i = start; i < sum.size(); ++i){
                int c = (start == patternLength + 1) ? (i - start) : (i - start + 1);
                if(zFunc[i] == patternLength)
                    std::cout << wordCoord[c].first << ", " << wordCoord[c].second << std::endl;
            }

            sum.erase(sum.begin() + patternLength + 1, sum.begin() + 2 * patternLength + 1);
            move(zFunc.begin() + patternLength + 1, zFunc.begin() + 2 * patternLength + 1, zFunc.begin());
            wordCoord.erase(wordCoord.begin(), wordCoord.begin() + patternLength);
            inPool = patternLength;
            start = patternLength + 2;
        }

    }

    ZPreprocess(sum, zFunc, patternLength + 1);
    for (int i = start; i < sum.size(); ++i)
    {
        int c = start == patternLength + 1 ? i - start : i - start + 1;
        if (zFunc[i] == patternLength)
            std::cout << wordCoord[c].first << ", " << wordCoord[c].second << std::endl;
    }

    return 0;
}

void ZPreprocess(std::vector<std::string> &str, std::vector<int> &z, int start)
{
    int len = str.size();
    int left = 0, right = 0;

    for (int cur = start; cur < len; ++cur){
        int count = 0;	// количество совпавших
        int compStart = 0;  // стартовая позиция для сравнения

        int img = cur - left;	// куда отобразится i-й элемент в своем z-блоке
        if (cur <= right)	// если i в каком-то z-блоке
            compStart = std::min(z[img], right - cur + 1);     // за старт принимаем минимальный из z[k] и длины остатка z-блока
                                                            // т.к. z[k] может выходить за границы z-блока, а там уже нужно сравнивать
        while (cur + count < len && str[count + compStart] == str[cur + count + compStart]) // ищем количество совпавших
            ++count;
        z[cur] = compStart + count;
        if (z[cur] > 0){	// если совпало больше 0, то нашелся z-блок, поэтому сдвигаем границы
            right = cur + z[cur] - 1;
            left = cur;
        }
    }
}


void StringToUpper(std::string &str){
    for (auto &c : str)
        c = toupper(c);
}
