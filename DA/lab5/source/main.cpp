#include <cassert>
#include <iostream>
#include <string>
#include <vector>

const int ALPHA_LEN = 27;

const char SENTINEL = '$';
const size_t INTERNAL = ((size_t)-1);

class TSuffTree {
   public:
    TSuffTree(const std::string &s) : str(std::move(s + SENTINEL)) {
        root = new TNode(0, 0, INTERNAL, nullptr);

        activeNode = root;
        activeEdge = 0;
        activeLength = 0;

        remain = 0;

        end = -1;
        suffNum = 0;

        size_t len = str.length();

        for (size_t i = 0; i < len; i++) {
            Phase(i);
        }
    }
    ~TSuffTree() {
        DeleteTree(root);
    }

    void Phase(size_t phaseNum) {
        remain++;
        end++;

        TNode *lastAddedInternalNode = nullptr;

        while (remain > 0) {
            if (activeLength == 0) {
                activeEdge = phaseNum;
            }
            if (activeNode->getEdge(str, activeEdge) == nullptr) {
                activeNode->getEdge(str, activeEdge) = new TNode(phaseNum, &end, suffNum, nullptr);

                suffNum++;
                remain--;

                if (lastAddedInternalNode != nullptr) {  
                    lastAddedInternalNode->suffLink = activeNode;
                    lastAddedInternalNode = nullptr;
                }
            } else {
                if (Walkdown())
                    continue;
                if (getNextCharAct() == str[phaseNum]) {
                    if (lastAddedInternalNode != nullptr && activeNode != root) {
                        lastAddedInternalNode->suffLink = activeNode;
                        lastAddedInternalNode = nullptr;
                    }
                    activeLength++;
                    break;
                } else {
                    TNode *toInsert = new TNode(phaseNum, &end, suffNum, nullptr);
                    suffNum++;

                    TNode *justInserted = Insert(toInsert);
                    if (lastAddedInternalNode != nullptr) {
                        lastAddedInternalNode->suffLink = justInserted;
                    }
                    lastAddedInternalNode = justInserted;
                    remain--;
                }
            }
            if (activeNode == root) {
                if (activeLength > 0) {
                    activeLength--;
                    activeEdge = phaseNum - remain + 1;
                }
            } else {
                activeNode = activeNode->suffLink;
            }
        }
    }

    void Print() {
        std::cout << "root" << std::endl;
        for (int i = 0; i < ALPHA_LEN; i++) {
            print(root->edges[i], 3);
        }
        std::cout << "suffix links: " << std::endl;
        for (int i = 0; i < ALPHA_LEN; i++) {
            print_suff_links(root->edges[i]);
        }
    }

    std::vector<int> getMatchStatistic(const std::string &text);
    std::vector<int> getMatchStatisticNaive(const std::string &text);

   private:
    struct TNode {
        TNode *edges[ALPHA_LEN];
        size_t left;
        size_t *right;
        size_t number;
        TNode *suffLink;
        TNode(size_t l, size_t *r, size_t n, TNode *node) : left(l), right(r), number(n), suffLink(node) {
            for (auto &el : edges) {
                el = nullptr;
            }
        }

        TNode *&getEdge(const std::string &str, size_t edge) {
            size_t c = (str[edge] == SENTINEL ? 26 : str[edge] - 'a');
            return edges[c];
        }

        size_t getLeft() {
            return left;
        }

        size_t getRight() {
            return number == INTERNAL ? (size_t)right : *right;
        }

        size_t getLength() {
            return getRight() - getLeft() + 1;
        }

        void Print(const std::string &str) {
            size_t rightBound = getRight();
            for (size_t i = left; i <= rightBound; i++) {
                std::cout << str[i];
            }
            if (number != INTERNAL) {
                std::cout << " - " << number;
            }
        }
    };

    std::string str;
    TNode *root;

    TNode *activeNode;
    size_t activeEdge;
    size_t activeLength;

    size_t end;
    size_t remain;
    size_t suffNum;

    int Walkdown() {
        TNode *edge = activeNode->getEdge(str, activeEdge);
        size_t len = edge->getLength();
        if (activeLength >= len) {
            activeLength -= len;
            activeEdge += len;
            activeNode = edge;
            return 1;
        }
        return 0;
    }

    char getNextCharAct() {
        size_t ind = activeNode->getEdge(str, activeEdge)->left + activeLength;
        return str[ind];
    }

    TNode *Insert(TNode *toInsert) {
        TNode *edge = activeNode->getEdge(str, activeEdge);
        size_t left = edge->getLeft();
        TNode *newInternalNode = new TNode(left, (size_t *)(left + activeLength - 1), INTERNAL, root);
        newInternalNode->getEdge(str, left + activeLength) = edge;
        edge->left = left + activeLength;
        newInternalNode->getEdge(str, toInsert->left) = toInsert;
        activeNode->getEdge(str, activeEdge) = newInternalNode;
        return newInternalNode;
    }

    void print(TNode *node, int tab) {
        const int tabIncr = 3;
        if (node == nullptr)
            return;
        for (int i = 0; i < tab; i++) {
            std::cout << ' ';
        }
        node->Print(str);
        std::cout << std::endl;
        for (int i = 0; i < ALPHA_LEN; i++) {
            print(node->edges[i], tab + tabIncr);
        }
    }

    void DeleteTree(TNode *node) {
        if (node == nullptr)
            return;
        if (node->number != INTERNAL) {
            delete node;
            return;
        }
        for (int i = 0; i < ALPHA_LEN; i++) {
            DeleteTree(node->edges[i]);
        }
        delete node;
    }

    void print_suff_links(TNode *node) {
        if (node == nullptr)
            return;
        if (node->suffLink != nullptr) {
            std::cout << "[ " << node->getLeft() << " , " << node->getRight() << " ]"
                      << " --> ";
            TNode *sl = node->suffLink;
            if (sl == root) {
                std::cout << "ROOT";
            } else if (sl == nullptr) {
                std::cout << "NULL";
            } else {
                std::cout << "[ " << sl->getLeft() << " , " << sl->getRight() << " ]";
            }
            std::cout << std::endl;
        }
        for (int i = 0; i < ALPHA_LEN; i++) {
            print_suff_links(node->edges[i]);
        }
    }
};

std::vector<int> TSuffTree::getMatchStatistic(const std::string &text) {
    size_t len = text.length();
    std::vector<int> result(len);

    TNode *lastNode = root;
    TNode *curNode;
    size_t curEdge = 0;
    size_t curLen = 0;

    size_t ind = 0;

    for (int i = 0; i < len; i++) {
        if (curLen == 0) {
            curNode = lastNode->getEdge(text, ind);
        } else {
            curNode = lastNode->getEdge(str, curEdge);
        }

        if (curNode != nullptr) {
            size_t curEdgeLen = curNode->getLength();
            while (curLen > curEdgeLen) {
                curEdge += curEdgeLen;
                curLen -= curEdgeLen;
                lastNode = curNode;
                curNode = curNode->getEdge(str, curEdge);
                curEdgeLen = curNode->getLength();
            }
            assert(curNode != nullptr);
            while (1) {
                if (curNode->getLength() == curLen) {
                    curLen = 0;
                    lastNode = curNode;
                    curNode = curNode->getEdge(text, ind);
                }
                if (curNode == nullptr || str[curNode->getLeft() + curLen] != text[ind]) {
                    break;
                }
                curLen++;
                ind++;
                if (ind >= len) {
                    for (int k = i; k < len; k++) {
                        result[k] = ind - k;
                    }
                    return result;
                }
            }
            if (curNode != nullptr) {
                curEdge = curNode->getLeft();
            }
        }

        result[i] = ind - i;

        if (lastNode == root) {
            if (curLen > 0) {
                curLen--;
                curEdge++;
            } else {
                ind++;
            }
        } else {
            lastNode = lastNode->suffLink;
        }
    }

    return result;
}

std::vector<int> TSuffTree::getMatchStatisticNaive(const std::string &text) {
    size_t len = text.length();
    std::vector<int> result(len);

    for (int i = 0; i < len; i++) {
        TNode *curNode = root->getEdge(text, i);
        if (curNode == nullptr) {
            result[i] = 0;
            continue;
        }
        size_t curLen = 0;
        size_t ind = i;
        while (1) {
            if (curNode->getLength() == curLen) {
                curLen = 0;
                curNode = curNode->getEdge(text, ind);
            }
            if (curNode == nullptr || str[curNode->getLeft() + curLen] != text[ind]) {
                break;
            }
            curLen++;
            ind++;
            if (ind >= len) {
                for (int k = i; k < len; k++) {
                    result[k] = ind - k;
                }
                return result;
            }
        }
        result[i] = ind - i;
    }
    return result;
}

int main() {
    std::string pattern;
    std::cin >> pattern;

    TSuffTree tree(pattern);

    std::string text;
    std::cin >> text;

    std::vector<int> ms = tree.getMatchStatistic(text);

    size_t patLen = pattern.length();
    size_t msSize = ms.size();
    for (size_t i = 0; i < msSize; i++) {
        if (ms[i] == patLen) {
            std::cout << i + 1 << std::endl;
        }
    }

    return 0;
}