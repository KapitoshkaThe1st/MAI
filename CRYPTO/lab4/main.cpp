#include <iostream>
#include <fstream>

using namespace std;

int file_size(ifstream &f){
    f.seekg (0, std::ios::end);
    unsigned int size = f.tellg();
    f.seekg (0, std::ios::beg);
    return size;
}

int main(int argc, char **argv){
    if(argc < 3){
        cout << "usage: prog <file1> <file2>" << endl;
        return 0;
    }

    ifstream f1(argv[1]), f2(argv[2]);
    if(!f1.is_open()){
        cout << "Can't open " << argv[1] << " file" << endl;
        return 0;
    }

    if(!f2.is_open()){
        cout << "Can't open " << argv[2] << " file" << endl;
        return 0;
    }

    unsigned int count_to_read = min(file_size(f1), file_size(f2));

    cout << "count to read: " << count_to_read << endl;

    unsigned int match_count = 0;
    for(unsigned int i = 0; i < count_to_read; ++i){
        if(f1.get() == f2.get())
            ++match_count;
    }

    cout << "match rate: " << (double)match_count / count_to_read << endl;
}