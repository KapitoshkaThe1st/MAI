#include <iostream>
#include <fstream>
#include <cstring>

#include "blake.h"

using namespace std;

int main(int argc, char **argv) {

    if(argc < 2){
        cout << "usage blake_hash <string>" << endl;
        return 0;
    }

    unsigned int hash[8];
    blake_hash(argv[1], strlen(argv[1]), hash);

    cout << hex;
    for (int i = 0; i < 8; ++i) {
        cout.width(8);
        cout.fill('0');
        cout << hash[i] << ' ';
    }
    cout << endl;
}