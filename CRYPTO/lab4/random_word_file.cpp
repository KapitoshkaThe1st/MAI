#include <iostream>
#include <fstream>
#include <cstdlib>

#include <vector>
#include <string>

using namespace std;

int main(int argc, char **argv){
    if(argc < 2){
        cout << "usage: prog <file1>" << endl;
        return 0;
    }

    ofstream f(argv[2]);

    if(!f.is_open()){
        cout << "Can't open " << argv[2] << " file" << endl;
        return 0;
    }

    ifstream wf("word_list");
    unsigned int word_count;
    wf >> word_count;

    vector<string> words(word_count);
    for(unsigned int i = 0; i < word_count; ++i)
        wf >> words[i];

    srand(time(0));

    unsigned int n = atoi(argv[1]);

    for(unsigned int i = 0; i < n; ++i){
        f << words[rand() % word_count] << ' ';
    }
    f << endl;
}