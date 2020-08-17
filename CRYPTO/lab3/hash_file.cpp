#include <iostream>
#include <fstream>

#include "blake.h"

using namespace std;

int main(int argc, char **argv) {

    if(argc < 2){
        cout << "usage blake_hash <file_to_hash>" << endl;
        return 0;
    }

    ifstream in(argv[1], std::fstream::binary);
    if(!in.is_open()){
        cout << "can't open file: " << argv[1] << endl;
    }

    unsigned int hash[8];
    blake_hash(in, hash);

    cout << hex;
    for (int i = 0; i < 8; ++i) {
        cout.width(8);
        cout.fill('0');
        cout << hash[i] << ' ';
    }
    cout << endl;
}