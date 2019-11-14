#include <iostream>
#include <string>
#include <fstream>
#include <cassert>

#include "helpers.h"

using namespace std;

int main(int argc, char **argv){

    char *inpFile = argv[1];
    char *outFile = argv[2];

    ifstream inp(inpFile);
    ofstream op(outFile);

    assert(inp);
    assert(op);

    cout << "started..." << endl;

    while(1){
        int lineCount;
        
        inp >> lineCount;
        inp.ignore((unsigned int)-1, '\n');

        if(inp.eof())
            break;
        
        op << lineCount << endl;

        string line;

        getline(inp, line);
        op << line << endl;
        for(int i = 0; i < lineCount - 1; ++i){
            getline(inp, line);
            op << line << endl;
        }

        getline(inp, line);
    }
    cout << "Finished!" << endl;
}
