#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <algorithm>
#include <set>
#include <cmath>

#include "helpers.h"

using namespace std;

typedef unordered_map<string, unsigned int> str_int_map; 
typedef unordered_map<string, str_int_map>  str_mp_map ;

#define DIFF_MULT 1.030 // score of proper tag is different less than 3% from the most likely tag's score

void getArgs(int c, char **v, string &inp, string &stat, string &outp){
    for(int i = 1; i < c; ++i){
        if(!strcmp(v[i], "--input")){
            inp = v[i + 1];
        }
        else if(!strcmp(v[i], "--stats")){
            stat = v[i + 1];
        }
        else if(!strcmp(v[i], "--output")){
            outp = v[i + 1];
        }
    }
}

int main(int argc, char **argv){

    string inpFile, statFile, outpFile;
    getArgs(argc, argv, inpFile, statFile, outpFile);
    
/*
    cout << "inp: " << inpFile << endl;
    cout << "stat: " << statFile << endl;
    cout << "outp: " << outpFile << endl;
*/

    if(!strcmp(argv[1], "learn")){
    
        // training
        unordered_set<string> toks;     // set of tokens for unique tokens counting 
        str_mp_map map;                 // main table token/class/count
        str_int_map classDocsCount;     // count of documents for each class
        str_int_map classToksCount;     // count of tokens for each class
        unsigned int docCount = 0;      // total documents count
        
        ifstream inp(inpFile);

        cout << "Training" << endl;

        while(!inp.eof()){
            // reading data

            // format:
            // <count of lines in document [n]> 
            // <document title>
            // <document main part>
            // <tag 1>,<tag 2>,...,<tag m>

            stringstream ss;        // for easier retrieving tokens from text of document
            vector<string> tags;    // vector of tags for current document
            int lineCount;

            inp >> lineCount;
            inp.ignore((unsigned int)-1, '\n');

            if(inp.eof())
                break;

            string line;

            // reading of text
            for(int i = 0; i < lineCount; ++i){
                getline(inp, line);
                ss << line << endl;
            }

            // reading of tags
            getline(inp, line);

            tags = split(line, ',');

            // for each tag increment count of docs
            for(auto &el : tags){
                classDocsCount[el]++;
            }

            // filling the map with frequencies
            string tok;
            while(ss >> tok){
                if((tok = filter(tok)) != ""){
                    toks.insert(tok);
                    for(auto &cls : tags){
                        classToksCount[cls]++;
                        map[tok][cls]++;
                    }
                }
            }
            docCount++;

            if(docCount % 100000 == 0){
                cout << "Processed: " << docCount << endl;
            }
        }

        cout << "Documents processed: " << docCount << endl;
        cout << "Training completed" << endl;
        
        // serialization
        cout << "Serialization" << endl;
        ofstream stat(outpFile);

        if(stat){
            // count of classes
            cout << "Count of classes: " << classDocsCount.size() << endl;

            stat << classDocsCount.size() << endl;
            // total count of docs for each class
            for(auto &el : classDocsCount){
                stat << el.first << ' ' << el.second << endl;
            }
            // total count of docs
            cout << "Count of documents: " << docCount << endl;

            stat << docCount << endl;

            // total count of unique tokens
            cout << "Count of unique tokens: " << toks.size() << endl;

            stat << toks.size() << endl;

            // count of classes
            stat << classToksCount.size() << endl;

            for(auto &el : classToksCount){
                stat << el.first << ' ' << el.second << endl;
            }

            // count of tokens
            stat << map.size() << endl;

            // for each token
            for(auto &tok : map){
                stat << tok.first;
                stat << endl;

                // count of classes for that class
                stat << tok.second.size() << endl;
                // for each class 
                for(auto &cls : tok.second){
                    // class and its count for that token
                    stat << cls.first << ' ' << cls.second << endl;
                }
            }
            cout << "Serialization completed" << endl;
        }
        else{
            cout << "Something went wrong" << endl;
            return 1;
        }
    }
    else{
        // deserialization
        str_mp_map map;
        unsigned int docCount = 0;
        unsigned int uniqueToksCount = 0;
        str_int_map classDocsCount;
        str_int_map classToksCount;

        vector<string> classes;

        ifstream stat(statFile);

        if(stat){

            cout << "Deserelization" << endl;
            
            // count of classes
            unsigned int countToRead;
            stat >> countToRead;

            cout << "Count of classes: " << countToRead << endl;

            // total count of docs for each class
            classDocsCount.reserve(countToRead);
            classes.reserve(countToRead);
            for(unsigned int i = 0; i < countToRead; ++i){
                string cls;
                int count;
                stat >> cls >> count;
                classDocsCount[cls] = count;
                classes.push_back(cls);
            }
            // total count of docs
            stat >> docCount;
            cout << "Count of documents: " << docCount << endl;

            // total count of unique tokens
            stat >> uniqueToksCount;
            cout << "Count of unique tokens: " << uniqueToksCount << endl;

            // count of classes
            stat >> countToRead;
            map.reserve(countToRead);

            // count of tokens in documents for each class
            for(unsigned int i = 0; i < countToRead; ++i){
                string cls;
                int count;
                stat >> cls >> count;
                classToksCount[cls] = count;
            }

            // count of classes
            stat >> countToRead;
            map.reserve(countToRead);

            // fill the table
            for(unsigned int i = 0; i < countToRead; ++i){
                string tok;
                int clsCount;
                stat >> tok >> clsCount;
                str_int_map &ref = map[tok];
                ref.reserve(clsCount);
                for(int j = 0; j < clsCount; ++j){
                    string cls;
                    int count;
                    stat >> cls >> count;
                    ref[cls] = count;
                }
            }
            cout << "Deserelization completed" << endl;
        }
        else{
            cout << "Something went wrong" << endl;
            return 1;
        }
        
        ifstream inp(inpFile);
        ofstream op(outpFile);

        if(op && inp){
            cout << "Classification" << endl;
            int docsProcessed = 0;
            while(1){
                stringstream ss;
                vector<string> tags;
                int lineCount;
                inp >> lineCount;

                if(inp.eof())
                    break;
                inp.ignore((unsigned int)-1, '\n');

                string line;
                
                // reading of text
                for(int i = 0; i < lineCount; ++i){
                    getline(inp, line);
                    ss << line << ' ';
                }

                vector<string> toks;

                string tok;

                // filter tokens
                while(ss >> tok){
                    if((tok = filter(tok)) != ""){
                        toks.push_back(tok);
                    }
                }

                // classification
                unsigned int size = classes.size();
                vector<double> results(size, 0.0);

                // for each class compute rate
                // formula: log(D_c/D) + sum for each token of log((W_ic + 1) / (|V| + L_c))
                // where: 
                // D_c -- count of docs of C class in training set
                // D -- total count of docs in training set
                // W_ic -- count of i-th token occurance in documents of class C 
                // |V| -- count of unique tokens
                // L_c -- total count of words of class C in training set
                // +1 -- laplace smoothing

                for(unsigned int i = 0; i < size; ++i){
                    string &cls = classes[i];
                    double &res = results[i];

                    // log(D_c / D)
                    res += log((double)classDocsCount[cls] / docCount);

                    // |V| + L_c
                    double denum = classToksCount[cls] + uniqueToksCount;

                    // summation
                    // for each token in main part
                    for(auto &tok : toks){
                        unsigned int count = 0;

                        auto externalIt = map.find(tok);
                        if(externalIt != map.end()){
                            auto nestedIt = externalIt->second.find(cls);
                            if(nestedIt != externalIt->second.end()){
                                count += nestedIt->second;
                            }
                        }

                        // W_ic + 
                        double num = count + 1;
                        double toAdd = log(num / denum);
                        res += toAdd;
                    }
                }

                vector<pair<double, string>> pairs;
                pairs.reserve(size);

                for(unsigned int i = 0; i < size; ++i){
                    pairs.push_back(make_pair(results[i], classes[i]));
                }

                // sorting by rate in ascending order
                sort(pairs.begin(), pairs.end(), std::greater<pair<double, string>>());

                // show and write first 5 of them or which differ less than 2.5% from maximum 
                int lim = 5;

                for(int i = 0; i < lim; ++i){
                    if(pairs[i].first > pairs[0].first * DIFF_MULT){
                        if(i > 0)
                            op << ',';
                        op << pairs[i].second;
                    }
                }
                op << endl;
                docsProcessed++;
            }
            cout << "Documents proessed: " << docsProcessed << endl;
            cout << "Classification completed" << endl;
        }
    }
}
