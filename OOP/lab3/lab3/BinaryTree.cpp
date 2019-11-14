#include "BinaryTree.h"
BinaryTree::node::node(std::shared_ptr<Figure> const &f) {
	obj = f;
	left = right = nullptr;
}
BinaryTree::BinaryTree() {
	m_root = nullptr;
}
BinaryTree::~BinaryTree() {
	m_root = del(m_root);
}
void BinaryTree::Insert(const std::shared_ptr<Figure> &f) {
	m_root = insert(m_root, new node(f));
}
void BinaryTree::Erase(const std::shared_ptr<Figure> &f) {
	m_root = remove(m_root, f);
}
std::shared_ptr<Figure> BinaryTree::Find(const std::shared_ptr<Figure> &f) {
	node* temp = find(m_root, f);
	if (temp)
		return temp->obj;
	else
		return std::shared_ptr<Figure>(nullptr);
	//return temp != nullptr ? temp->obj : std::shared_ptr<Figure>(nullptr));
}
void BinaryTree::Print() {
	print(std::cout, m_root, 0);
}
BinaryTree::node* BinaryTree::insert(node* tree, node* fresh) {
	if (!tree)
		return fresh;
	if (*(fresh->obj) < *(tree->obj))
		tree->left = insert(tree->left, fresh);
	else
		tree->right = insert(tree->right, fresh);
	return tree;
}
BinaryTree::node* BinaryTree::find(node* tree, const std::shared_ptr<Figure> &f) {
	while (tree != nullptr)
		if (*f < *(tree->obj))
			tree = tree->left;
		else if (*f > *(tree->obj))
			tree = tree->right;
		else
			break;
	return tree;
}
BinaryTree::node* BinaryTree::remove(node *tree, const std::shared_ptr<Figure> &f) {
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
			node *m = unlink_min(tree->right);
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
BinaryTree::node* BinaryTree::unlink_min(node *tree) {
	node* parent = nullptr;
	while (tree->left) {
		parent = tree;
		tree = tree->left;
	}
	if (parent) {
		parent->left = tree->right;
	}
	return tree;
}
void BinaryTree::print(std::ostream &os, node *tree, int tab) {
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
BinaryTree::node* BinaryTree::del(node* tree) {
	if (!tree)
		return nullptr;
	tree->left = del(tree->left);
	tree->right = del(tree->right);
	delete tree;
	return nullptr;
}
std::ostream& operator<< (std::ostream &os, BinaryTree &bt) {
	bt.print(os, bt.m_root, 0);
	return os;
}