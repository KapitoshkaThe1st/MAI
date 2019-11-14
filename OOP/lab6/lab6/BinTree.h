#ifndef BIN_TREE_H
#define BIN_TREE_H

#include "Iterator.h"

template<typename T>
struct node {
	std::shared_ptr<T> obj;
	node<T> *left, *right, *parent;
	node(T*);
};

template<typename T>
node<T>::node(T *obj_ptr) {
	obj = std::shared_ptr<T>(obj_ptr);
	left = right = parent = nullptr;
}

template<typename T>
class BinTree {
	friend class Iterator<T>;
	friend std::ostream& operator<<(std::ostream &os, BinTree<T> &bt) {
		bt.print(os, bt.m_root, 0);
		return os;
	}
	typedef node<T>* node_ptr;
public:
	BinTree();
	~BinTree();
	void Insert(T *obj_ptr);
	void Remove(T&);
	void Print();
	Iterator<T> Find(T&);
	void Clear();
	Iterator<T> begin();
	Iterator<T> end();
private:
	node_ptr m_root;
	void insert(node_ptr, node_ptr);
	void remove(node_ptr, T&);
	void print(std::ostream&, node_ptr, int);
	node_ptr find(node_ptr, T&);
	static node_ptr min(node_ptr);
	static node_ptr max(node_ptr);
	static node_ptr next(node_ptr);
	static node<T>* prev(node_ptr tree, BinTree<T>* bin_tree_ptr);
	void clear(node_ptr);
};

template<typename T>
BinTree<T>::BinTree() : m_root(nullptr) {}

template<typename T>
BinTree<T>::~BinTree() {
	clear(m_root);
}

template<typename T>
void  BinTree<T>::Insert(T *obj_ptr) {
	insert(m_root, new node<T>(obj_ptr));
}

template<typename T>
void BinTree<T>::Print() {
	print(std::cout, m_root, 0);
}

template<typename T>
Iterator<T> BinTree<T>::Find(T &obj){
	return Iterator<T>(find(m_root, obj), this);
}

template<typename T>
void BinTree<T>::Remove(T &obj) {
	remove(m_root, obj);
}

template<typename T>
void BinTree<T>::Clear() {
	clear(m_root);
	m_root = nullptr;
}

template<typename T>
Iterator<T> BinTree<T>::begin() {
	return Iterator<T>(min(m_root), this);
}

template<typename T>
Iterator<T> BinTree<T>::end() {
	return Iterator<T>(nullptr, this);
}

template<typename T>
void BinTree<T>::insert(node_ptr tree, node_ptr ins) {
	if (m_root == nullptr) {
		m_root = ins;
		m_root->parent = nullptr;
		return;
	}
	node_ptr parent = nullptr;
	while (tree != nullptr) {
		parent = tree;
		if (*(ins->obj) < *(tree->obj))
			tree = tree->left;
		else
			tree = tree->right;
	}
	if (*(ins->obj) < *(parent->obj)) {
		parent->left = ins;
		ins->parent = parent;
	}
	else {
		parent->right = ins;
		ins->parent = parent;
	}
}

template<typename T>
void BinTree<T>::remove(node_ptr tree, T &obj) {
	bool is_left_side = true;
	node_ptr parent = nullptr;

	while (!(*(tree->obj) == obj)) {
		parent = tree;
		if (obj < *(tree->obj)) {
			is_left_side = true;
			tree = tree->left;
		}
		else if (obj > *(tree->obj)) {
			is_left_side = false;
			tree = tree->right;
		}
		if (tree == nullptr)
			return;
	}

	if (parent == nullptr) {	// удаление корня
		node_ptr minimal = min(m_root->right);
		if (m_root->right == minimal) {
			minimal->left = m_root->left;
			if (minimal->left != nullptr)
				minimal->left->parent = m_root;
		}
		else {
			minimal->parent->left = minimal->right; // отвязка минимального
			if (minimal->right != nullptr)
				minimal->right->parent = minimal->parent;
			minimal->left = m_root->left;	// вставка на место корня
			if (minimal->left != nullptr)
				minimal->left->parent = minimal;
			minimal->right = m_root->right;
			if (minimal->right != nullptr)
				minimal->right->parent = minimal;
		}
		minimal->parent = nullptr;
		delete m_root;
		m_root = minimal;
		return;

	}
	if (tree->left == nullptr && tree->right == nullptr) {
		if (is_left_side)
			parent->left = nullptr;
		else
			parent->right = nullptr;
	}
	else if (tree->left == nullptr) {
		if (is_left_side)
			parent->left = tree->right;
		else
			parent->right = tree->right;
	}
	else if (tree->right == nullptr) {
		if (is_left_side)
			parent->left = tree->left;
		else
			parent->right = tree->left;
	}
	else {
		node_ptr minimal = min(tree->right);
		if (tree->right == minimal) {
			minimal->left = tree->left;
			minimal->parent = parent;
			if (is_left_side) {
				parent->left = minimal;
			}
			else {
				parent->right = minimal;
			}
		}
		else {
			minimal->parent->left = minimal->right; // отвязка минимального
			if (minimal->right != nullptr)
				minimal->right->parent = minimal->parent;
			minimal->parent = parent;	// вставка на место удаляемого
			minimal->left = tree->left;
			if (minimal->left != nullptr)
				minimal->left->parent = minimal;
			minimal->right = tree->right;
			if (minimal->right != nullptr)
				minimal->right->parent = minimal;
			if (is_left_side) {
				parent->left = minimal;
			}
			else {
				parent->right = minimal;
			}
		}
	}
	delete tree;
	return;
}

template<typename T>
void BinTree<T>::print(std::ostream &os, node_ptr tree, int tab) {
	const int tab_increment = 2;
	if (tree == nullptr)
		return;
	print(os, tree->left, tab + tab_increment);
	for (int i = 0; i < tab; ++i)
		os << ' ';
	os << *(tree->obj) << endl;
	print(os, tree->right, tab + tab_increment);
}

template<typename T>
node<T>* BinTree<T>::find(node_ptr tree, T &obj) {
	while (tree != nullptr && !(*(tree->obj) == obj)) {
		if (obj < *(tree->obj))
			tree = tree->left;
		else
			tree = tree->right;
	}
	return tree;
}

template<typename T>
node<T>* BinTree<T>::min(node_ptr tree) {
	if (tree == nullptr)
		return nullptr;
	while (tree->left != nullptr)
		tree = tree->left;
	return tree;
}

template<typename T>
node<T>* BinTree<T>::max(node_ptr tree) {
	if (tree == nullptr)
		return nullptr;
	while (tree->right != nullptr)
		tree = tree->right;
	return tree;
}

template<typename T>
node<T>* BinTree<T>::next(node_ptr tree) {
	if (tree->right != nullptr)
		return min(tree->right);
	while (tree->parent != nullptr && tree->parent->left != tree) {
		tree = tree->parent;
	}
	return tree->parent;
}

template<typename T>
node<T>* BinTree<T>::prev(node_ptr tree, BinTree<T> *bin_tree_ptr) {
	if (tree == nullptr)
		return max(bin_tree_ptr->m_root);
	if (tree->left != nullptr)
		return max(tree->left);
	while (tree->parent != nullptr && tree->parent->right != tree) {
		tree = tree->parent;
	}
	return tree->parent;
}

template<typename T>
void BinTree<T>::clear(node_ptr tree) {
	if (tree == nullptr)
		return;
	clear(tree->left);
	clear(tree->right);
	delete tree;
}
#endif // !BIN_TREE_H