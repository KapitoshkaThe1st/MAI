#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include <cassert>

using namespace std;


struct Tree{
    int freq;
    char symb;
    Tree *left;
    Tree *right;
    Tree(int f, char s, Tree *l = nullptr, Tree *r = nullptr) : freq(f), symb(s), left(l), right(r) {}
    ~Tree() = default;
};

// class TreeLeaf{
//     char symb;
// }

// const vector<pair<int, char>> &freqs

bool comp(Tree *a, Tree *b){
    return a->freq > b->freq;
}

Tree* makeHuffmanTree(const unordered_map<char, int> &freqs){
    int n = freqs.size();

    vector<Tree*> pq;
    pq.reserve(n);

    for(auto &el : freqs){
        pq.push_back(new Tree(el.second, el.first));
        // Tree *tmp = pq.back();
        // cout << tmp->symb << " - " << tmp->freq << endl;
    }

    make_heap(pq.begin(), pq.end(), comp);

    for(int i = 0; i < n - 1; ++i){
        Tree *left = pq.front();
        pop_heap(pq.begin(), pq.end(), comp);
        pq.pop_back();

        Tree *right = pq.front();
        pop_heap(pq.begin(), pq.end(), comp);
        pq.pop_back();
        
        Tree* temp = new Tree(left->freq + right->freq, 0, left, right);

        pq.push_back(temp);
        push_heap(pq.begin(), pq.end(), comp);
    }

    assert(pq.size() == 1);

    return pq.front();
}

void getCodeTable(const Tree *node, string code, unordered_map<char, string> &table){
    if(node->left == nullptr || node->right == nullptr){
        table[node->symb] = code;
        return;
    }

    getCodeTable(node->left, code + '0', table);
    getCodeTable(node->right, code + '1', table);
}

void deleteHuffmanTree(const Tree *node){
    if(node == nullptr){
        return;
    }

    deleteHuffmanTree(node->left);
    deleteHuffmanTree(node->right);

    delete node;
}

void printHuffmanTree(const Tree *node, int tab = 0){
    if(node == nullptr){
        return;
    }

    printHuffmanTree(node->right, tab + 3);
    for(int i = 0; i < tab; ++i){
        cout << ' ';
    }
    cout << "(" << node->symb << ", " << node->freq << ")" << endl;
    printHuffmanTree(node->left, tab + 3);

}

string HuffmanDecode(Tree *tree, const string &code){
    int ind = 0;
    Tree *node = tree;

    int len = code.length();


    string res;
    while(ind < len){
        while(node->left != nullptr && node->right != nullptr){
            if(code[ind] == '0'){
                node = node->left;
            }
            else{
                node = node->right;
            }
            ind++;
        }
        res.push_back(node->symb);
        node = tree;
    }
    
    return res;
}

int main(){

    int n;
    cin >> n;
    
    unordered_map<char, int> freqs;

    for(int i = 0; i < n; ++i){
        char symb;
        cin >> symb >> freqs[symb];
    }

    // for(auto &el : freqs){
    //     cout << el.first << " - " << el.second << endl;
    // }

    Tree *ht = makeHuffmanTree(freqs);
    printHuffmanTree(ht);

    unordered_map<char, string> codeTable;
    getCodeTable(ht, "", codeTable);

    for(auto &el : codeTable){
        cout << el.first << " - " << el.second << endl;
    }

    cout << HuffmanDecode(ht, "1011001100111100") << endl;;
    
    deleteHuffmanTree(ht);
}

// bool comp(int a, int b){
//     return a > b;
// }

// int main(){
//     vector<int> heap = {4, 3, 1, 7, 43, 65, 2, 3, 6};
//     make_heap(heap.begin(), heap.end(), comp);

//     for(auto &el : heap){
//         cout << el << ' ';
//     }
//     cout << endl;

//     cout << "head: " << heap.front() << endl;

//     pop_heap(heap.begin(), heap.end(), comp);
//     heap.pop_back();

//     for(auto &el : heap){
//         cout << el << ' ';
//     }
//     cout << endl;


//     heap.push_back(1488);
//     push_heap(heap.begin(), heap.end(), comp);

//     cout << "head: " << heap.front() << endl;
//     for(auto &el : heap){
//         cout << el << ' ';
//     }
//     cout << endl;

// }
