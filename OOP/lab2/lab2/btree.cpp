#include "btree.hpp"

node* Tree::min(node *tree) {
	if (tree == nullptr) {
		return nullptr;
	}
	while (tree->left != nullptr) {
		tree = tree->left;
	}
	return tree;
}
node* Tree::max(node *tree) {
	if (tree == nullptr) {
		return nullptr;
	}
	while (tree->right != nullptr) {
		tree = tree->right;
	}
	return tree;
}
void Tree::print(std::ostream &os, int tab, node *tree) const {
	if (tree == nullptr) {
		return;
	}
	print(os, tab + 2, tree->left);
	for (int i = 0; i < tab; ++i) {
		os << ' ';
	}
	os << tree->data << std::endl;
	print(os, tab + 2, tree->right);
}
void Tree::remove(node* tree) {
	if (tree == nullptr)
		return;
	remove(tree->left);
	remove(tree->right);
	if (tree->parent != nullptr) {
		if (tree->parent->data > tree->data) {
			tree->parent->left = nullptr;
		}
		else {
			tree->parent->right = nullptr;
		}
	}
	delete tree;
}
Tree::Tree() : root(nullptr) { }
Tree::~Tree() {
	remove(root);
}

void Tree::Add(Rhomb val) {
	node *ins = new node();
	ins->data = val;
	ins->parent = ins->left = ins->right = nullptr;

	node *temp = root;
	node *parent = nullptr;

	while (temp != nullptr) {
		parent = temp;
		if (temp->data < val) {
			temp = temp->right;
		}
		else {
			temp = temp->left;
		}
	}
	ins->parent = parent;
	if (parent == nullptr) {
		root = ins;
	}
	else if (parent->data > val) {
		parent->left = ins;
	}
	else {
		parent->right = ins;
	}
}
void Tree::Remove(Rhomb val) {
	if (root == nullptr)
		return;
	node *temp = root;
	node *parent = nullptr;

	while (!(temp->data == val)) {

		parent = temp;
		if (temp->data < val) {
			temp = temp->right;
		}
		else {
			temp = temp->left;
		}
		if (temp == nullptr)
			return;
	}

	if (temp->right == nullptr) {
		// (1) если нет правого потомка -- просто удаляем элемент, замещая левым предком 
		if (parent != nullptr) {
			if (parent->data > temp->data)
				parent->left = temp->left;
			else
				parent->right = temp->left;
		}
		else {
			root = temp->left;
		}
		delete temp;
	}
	else if (temp->left == nullptr) {
		// (2) если нет левого потомка -- просто удаляем элемент, замещая правым предком
		if (parent != nullptr) {
			if (parent->data > temp->data)
				parent->left = temp->right;
			else
				parent->right = temp->right;
		}
		else {
			root = temp->right;
		}
		delete temp;
	}
		// случай с отсутствием обоих поддеревьев автоматически учитывается одним из (1) или (2).
	else {
		// (3) если есть оба потомка, то ищем следующий за текущим элемент. Он либо минимальный в правом поддереве,
		//тогда заменяем текущий на него, потомки текущего отдаем его предку, либо он не в правом поддереве, а один из потомков,
		//но такое может быть только в том случае, когда нет правого поддерева, и тогда этот случай вырождается в случай (1).
		node *m = min(temp->right);
		if (parent != nullptr) {
			if (parent->data > temp->data)
				parent->left->data = m->data;
			else
				parent->right->data = m->data;
		}
		else {
			root->data = m->data;
		}
		if (temp->right == m)
			m->parent->right = m->right;
		else
			m->parent->left = m->right;
		delete m;
	}
}

node* Tree::Find(Rhomb val) {
	if (root == nullptr)
		return nullptr;
	node *temp = root;
	while (!(temp->data == val)) {
		if (temp->data > val) {
			temp = temp->left;
		}
		else{
			temp = temp->right;
		}
		if (temp == nullptr) {
			return nullptr;
		}
	}
	return temp;
}

Rhomb Tree::Min() {
	node *temp = min(root);
	return temp != nullptr ? temp->data : Rhomb(-1.0, -1.0);
}
Rhomb Tree::Max() {
	node *temp = max(root);
	return temp != nullptr ? temp->data : Rhomb(-1.0, -1.0);
}
void Tree::Print() {
	print(std::cout, 0, root);
}
std::ostream& operator<<(std::ostream &os, const Tree& tree) {
	tree.print(os, 0, tree.root);
	return os;
}