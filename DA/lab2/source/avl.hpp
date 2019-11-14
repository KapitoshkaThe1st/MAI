#ifndef AVL_H
#define AVL_H

#include <fstream>
#include <iostream>
#include <cstring>

template<typename K, typename T>
class TAVL
{
private:

	struct node{
		K key;
		T value;
		node *left, *right;
		short height;
		node(const K &k, const T &v){
			key = k;
			value = v;
			left = right = nullptr;
			height = 1;
		}
	};

	node *root;

	node* find(node* tree,const K &k){
		while(tree != nullptr){
			if(k > tree->key){
				tree = tree->right;
			}
			else if(k < tree->key){
				tree = tree->left;
			}
			else{
				return tree;	
			}
		}
		return tree;
	}

	node* insert(node *tree, node *ins){
		if(tree == nullptr){
			return ins;
		}
		if(ins->key < tree->key){
			node *temp = insert(tree->left, ins);
			if(temp == nullptr)
				return nullptr;
			tree->left = temp;
		}
		else if (ins->key > tree->key){
			node *temp = insert(tree->right, ins);
			if(temp == nullptr)
				return nullptr;
			tree->right = temp;
		}
		else{
			return nullptr;
		}
		reheight(tree);		
		return balance(tree);
	}

	node* remove(node *tree, const K &key){
		if(tree == nullptr){
			return nullptr;
		}
		if(key < tree->key){
			node *temp = remove(tree->left, key);
			tree->left = temp;
		}	
		else if(key > tree->key){
			node *temp = remove(tree->right, key);			
			tree->right = temp;
		}
		else{
			if(tree->right == nullptr){
				node *temp = tree->left;
				delete tree;
				return temp;
			}
			else if(tree->left == nullptr){
				node *temp = tree->right;
				delete tree;
				return temp;
			}
			else{
				node *m = min(tree->right);
				tree->key = m->key;
				tree->value = m->value;
				tree->right = removeMin(tree->right);
				delete m;
			}
		}
		reheight(tree);
		return balance(tree);
	}

	void print(node *tree, int tab){
		const int TAB_INCR = 4;
		if(tree == nullptr){
			return;
		}
		print(tree->right, tab + TAB_INCR);
		for(int i = 0; i < tab; ++i){
			std::cout << ' ';
		}
		std::cout << 
		// tree->key <<
		"< " << tree->key  << ", " << tree->value << " >" <<
		// "(h:" << tree->height << ")" <<
		// "(bf:" << bFactor(tree) << ")" <<
		std::endl;
		print(tree->left, tab + TAB_INCR);
	}

	node* min(node *tree){
		while(tree->left != nullptr){
			tree = tree->left;
		}
		return tree;
	}

	node* removeMin(node *tree){
		if(tree->left == nullptr){
			return tree->right;
		}
		tree->left = removeMin(tree->left);
		return balance(tree);
	}

	node* rrotate(node *tree){
		node *temp = tree->left;
		tree->left = temp->right;
		temp->right = tree;
		reheight(tree);
		reheight(temp);
		return temp; 
	}

	node* lrotate(node *tree){
		node *temp = tree->right;
		tree->right = temp->left;
		temp->left = tree;
		reheight(tree);
		reheight(temp);
		return temp; 
	}

	node* balance(node *tree){
		short bf = bFactor(tree);
		if(bf < -1){
			if(bFactor(tree->left) > 0){
				tree->left = lrotate(tree->left);
			}
			tree = rrotate(tree);
		}else if(bf > 1){
			if(bFactor(tree->right) < 0){
				tree->right = rrotate(tree->right);
			}
			tree = lrotate(tree);
		}
		reheight(tree);
		return tree;
	}

	inline short max(short a, short b){
		return a > b ? a : b;
	}

	inline short height(node *tree){
		return tree == nullptr ? 0 : tree->height;
	}

	void reheight(node *tree){
		tree->height = max(height(tree->left), height(tree->right)) + 1;
	}

	inline short bFactor(node *tree){
		return height(tree->right) - height(tree->left);
	}

	void deleteTree(node* tree){
		if(tree == nullptr)
			return;
		deleteTree(tree->left);
		deleteTree(tree->right);
		delete tree;
		return;
	}

	int isAVL(node* tree){
		if(tree == nullptr){
			return 1;
		}
		if(isAVL(tree->left) == 0 || isAVL(tree->right) == 0)
			return 0;
		if(bFactor(tree) <= 1 && bFactor(tree) >= -1)
			return 1;
		else
			return 0;
	}

	int serialize(node *tree, std::ofstream &ofs){
		if(tree == nullptr){
			return 1;
		}
		int res = 1;
		res &= serialize(tree->left, ofs);
		ofs << tree->key;
		ofs.write((char*)&(tree->value), sizeof(tree->value));
		res &= serialize(tree->right, ofs);
		return res;
	}

public:
	void Swap(TAVL<K,T> &other){
		node *temp = root;
		root = other.root;
		other.root = temp;
	}
	int Deserialize(std::ifstream &inp){
		T val = T();
		K str = K();
		const char *secret_sequence = "it'stree";
		char buffer[9] = {0};
		inp.read(buffer, strlen(secret_sequence) * sizeof(char));
		if(strcmp(buffer, secret_sequence) != 0)
			return 0;
		while(1){
			inp >> str;
			inp.read((char*)&val, sizeof(val));
			if(inp.gcount() != sizeof(val))
				break;
			Insert(str, val);
		}
		return 1;
	}

	int Serialize(std::ofstream &ofs){
		const char *secret_sequence = "it'stree";
		ofs.write(secret_sequence, strlen(secret_sequence) * sizeof(char));
		return serialize(root, ofs);
	}

	T* Find(const K &k){
		node *temp = find(root, k);
		return temp == nullptr ? nullptr : &(temp->value);
	}

	TAVL(){
		root = nullptr;	
	}
	~TAVL(){
		deleteTree(root);
	}

	int Insert(const K &k, const T &v){
		if(find(root, k) != nullptr){
			return 0;
		}
		root = insert(root, new node(k, v));
		return 1;
	}
	int Remove(const K &k){
		if(find(root, k) == nullptr){
			return 0;
		}
		root = remove(root, k);
		return 1;
	}

	int Empty(){
		return root == nullptr;
	}

	void Print(){
		print(root, 0);
	}
};
#endif