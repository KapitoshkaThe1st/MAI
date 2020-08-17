#include <iostream>
#include <fstream>
#include <cstring>

#include "blake.h"

using namespace std;

#define MAX_ROUND 20

unsigned int compare(unsigned int a, unsigned int b){
    unsigned int sum = 0;
    unsigned int t = a ^ b;
    for(int i = 0; i < 32; ++i)
        sum += (t >> i) & 1;
    return sum;
}

double strong_avalanche_effect(const string &s, int round_count){
    int n = s.length();
    unsigned int hash[8]; 
    unsigned int hash1[8];

    vector<double> v(32 * 8, 0);

    for(int i = 0; i < n * 8; ++i){
        string s1 = s;
        s1[i / 8] ^= (1 << (i % 8));

        blake_hash(s.c_str(), n, hash, round_count);
        blake_hash(s1.c_str(), n, hash1, round_count);

        for(int j = 0; j < 8; ++j){
            unsigned int t = hash[j] ^ hash1[j];
            for(int k = 0; k < 32; ++k){
                if((t >> k) & 1){
                    v[j*32 + k]++;
                }
            }
        }
    }
    double sum = 0.0;
    for(int i = 0; i < 32*8; ++i){
        v[i] /= n * 8;
        sum += v[i];
    }
    return sum / (32 * 8);
}

void bits_changed(const string &s){
    int n = s.length();

    unsigned int hash[8];
    unsigned int hash1[8];
    
    vector<double> c(MAX_ROUND + 1, 0);

    for(int k = 0; k < n; ++k){
        string s1 = s;
        s1[k] ^= 1;

        for(int i = 1; i <= MAX_ROUND; ++i){
            blake_hash(s.c_str(), n, hash, i);
            blake_hash(s1.c_str(), n, hash1, i);
            
            int count = 0;

            for(int j = 0; j < 8; ++j){
                count += compare(hash[j], hash1[j]);
            }
            c[i] += count;
        }
    }

    cout << "i = [";
    for(int i = 1; i < MAX_ROUND + 1; ++i){
        if(i > 1)
            cout << ", ";
        cout << i;
    }
    cout << "]" << endl;

    cout << "c = [";
    for(int i = 1; i < MAX_ROUND + 1; ++i){
        if(i > 1)
            cout << ", ";
        cout << c[i] / n;
    }
    cout << "]" << endl;
}

int main() {
    string s = "ggfhfdghehteswhngfhfgbfhfgjgkhgcfhdgesgrhrtjhtydrthrthbrtefbervhbleigbvilegirebglergvbeliagrkdjbfdljsnldslsdfnegvaebasergkelragae";
    
    bits_changed(s);

    vector<double> p(MAX_ROUND+1, 0);
    for(int i = 1; i <= MAX_ROUND; ++i){
        p[i] = strong_avalanche_effect(s, i);
    }

    cout << "p = [";
    for(int i = 1; i < MAX_ROUND + 1; ++i){
        if(i > 1)
            cout << ", ";
        cout << p[i];
    }
    cout << "]" << endl;
}