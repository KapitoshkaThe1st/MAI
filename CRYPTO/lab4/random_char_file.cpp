#include <iostream>
#include <fstream>
#include <cstdlib>

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

    srand(time(0));

    unsigned int n = atoi(argv[1]);

    for(unsigned int i = 0; i < n; ++i){
        unsigned int c = rand() % 52;
        f.put(c < 26 ? 'a' + c : 'A' + c % 26);
    }
}