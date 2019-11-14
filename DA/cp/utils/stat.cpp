#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <functional>

#include "helpers.h"

using namespace std;

int main(int argc, char **argv){
    ifstream inp(argv[1]);

    unordered_map<string, unsigned int> tagCount;

    unsigned int docCount = 0;
    while(1){
        string line;
        
        int lineCount;
        inp >> lineCount;
        
        if(inp.eof())
            break;
        
        inp.ignore((unsigned int) -1, '\n');

        for(int i = 0; i < lineCount; ++i){
            getline(inp, line);
        }
        getline(inp, line);

        vector<string> tags = split(line, ',');

        for(auto &tag : tags){
            tagCount[tag]++;
        }

        docCount++;
    }


    vector<pair<double, string>> freqs;
    freqs.reserve(tagCount.size());

    for(auto &el : tagCount){
        // cout << el.first << ' ' << (double)el.second / docCount << endl;
        double freq = (double)el.second / docCount;
        freqs.push_back(make_pair(freq, el.first));
    } 

    using dspair = pair<double, string>;

    sort(freqs.begin(), freqs.end(), greater<dspair>());

    for(auto &el : freqs){
        cout << el.second << " : " << el.first << endl;
    }
}