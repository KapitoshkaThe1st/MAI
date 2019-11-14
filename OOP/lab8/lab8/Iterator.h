#ifndef ITERATOR_H
#define ITERATOR_H

#include <memory>

#include "BinTree.h"

template<typename T>
struct node;

template<typename T>
class BinTree;

template<typename T>
class Iterator {
	typedef node<T>* node_ptr;
private:
	BinTree<T> *bin_tree_ptr;
	node_ptr tree;
public:
	Iterator() {};
	Iterator(node_ptr p, BinTree<T> *bt_ptr) : tree(p), bin_tree_ptr(bt_ptr) {}
	std::shared_ptr<T>& operator*() {
		return tree->obj;
	}
	T* operator->() {
		return tree->obj.get();
	}
	Iterator<T>& operator++() {
		tree = BinTree<T>::next(tree);
		return *this;
	}
	Iterator<T>& operator--() {
		tree = BinTree<T>::prev(tree, bin_tree_ptr);
		return *this;
	}
	bool operator==(const Iterator<T> &other) {
		return bin_tree_ptr == other.bin_tree_ptr && tree == other.tree;
	}
	bool operator!=(const Iterator<T> &other) {
		return !(*this == other);
	}
};
#endif // !ITERATOR_H