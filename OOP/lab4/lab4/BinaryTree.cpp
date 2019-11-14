#include "BinaryTree.h"
template<typename T>
node<T>::node(std::shared_ptr<T> const &f) {
	obj = f;
	left = right = nullptr;
}
template<typename T>
BinaryTree<T>::BinaryTree() {
	m_root = nullptr;
}
template<typename T>
BinaryTree<T>::~BinaryTree() {
	m_root = del(m_root);
}
template<typename T>
void BinaryTree<T>::Insert(const std::shared_ptr<T> &f) {
	m_root = insert(m_root, new node<T>(f));
}
template<typename T>
void BinaryTree<T>::Erase(const std::shared_ptr<T> &f) {
	m_root = remove(m_root, f);
}
template<typename T>
std::shared_ptr<T> BinaryTree<T>::Find(const std::shared_ptr<T> &f) {
	node<T>* temp = find(m_root, f);
	if (temp)
		return temp->obj;
	else
		return std::shared_ptr<T>(nullptr);
	//return temp != nullptr ? temp->obj : std::shared_ptr<Figure>(nullptr));
}
template<typename T>
void BinaryTree<T>::Print() {
	print(std::cout, m_root, 0);
}
template<typename T>
node<T>* BinaryTree<T>::insert(node<T>* tree, node<T>* fresh) {
	if (!tree)
		return fresh;
	if (*(fresh->obj) < *(tree->obj))
		tree->left = insert(tree->left, fresh);
	else
		tree->right = insert(tree->right, fresh);
	return tree;
}
template<typename T>
node<T>* BinaryTree<T>::find(node<T>* tree, const std::shared_ptr<T> &f) {
	while (tree != nullptr)
		if (*f < *(tree->obj))
			tree = tree->left;
		else if (*f > *(tree->obj))
			tree = tree->right;
		else
			break;
	return tree;
}
template<typename T>
node<T>* BinaryTree<T>::remove(node<T> *tree, const std::shared_ptr<T> &f) {
	if (tree == nullptr)
		return tree;
	if (*f < *(tree->obj))
		tree->left = remove(tree->left, f);
	else if (*f > *(tree->obj))
		tree->right = remove(tree->right, f);
	else {
		if (!tree->left)
			return tree->right;
		else if (!tree->right)
			return tree->left;
		else {
			node<T> *m = unlink_min(tree->right);
			tree->obj = m->obj;
			if (tree->right == m) {
				tree->right = m->right;
			}
			else {
				tree->right = m->right;
				tree->right->left = m->left;
			}
		}
	}
	return tree;
}
template<typename T>
node<T>* BinaryTree<T>::unlink_min(node<T> *tree) {
	node<T>* parent = nullptr;
	while (tree->left) {
		parent = tree;
		tree = tree->left;
	}
	if (parent) {
		parent->left = tree->right;
	}
	return tree;
}
template<typename T>
void BinaryTree<T>::print(std::ostream &os, node<T> *tree, int tab) {
	const int tab_incr = 2;
	if (!tree)
		return;
	print(os, tree->left, tab + tab_incr);
	for (int i = 0; i < tab; ++i)
		os << ' ';
	os << *(tree->obj)/* << std::endl*/;
	//tree->obj->Print(os);
	print(os, tree->right, tab + tab_incr);
}
template<typename T>
node<T>* BinaryTree<T>::del(node<T>* tree) {
	if (!tree)
		return nullptr;
	tree->left = del(tree->left);
	tree->right = del(tree->right);
	delete tree;
	return nullptr;
}
template<typename T>
std::ostream& operator<< (std::ostream &os, BinaryTree<T> &bt) {
	bt.print(os, bt.m_root, 0);
	return os;
}