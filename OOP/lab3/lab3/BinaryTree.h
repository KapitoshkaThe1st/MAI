#ifndef BINTREE_H
#define BINTREE_H

#include <memory>
#include <iostream>

#include "Figure.h"

class BinaryTree {
public:
	struct node {
		std::shared_ptr<Figure> obj;
		node *left, *right;
		node(std::shared_ptr<Figure> const&);
	};
	BinaryTree();
	~BinaryTree();
	void Insert(const std::shared_ptr<Figure>&);
	void Erase(const std::shared_ptr<Figure>&);
	std::shared_ptr<Figure> Find(const std::shared_ptr<Figure>&);
	void Print();
private:
	node *m_root;
	node* insert(node*, node*);
	node* find(node*, const std::shared_ptr<Figure>&);
	node* remove(node*, const std::shared_ptr<Figure>&);
	node* unlink_min(node*);
	void print(std::ostream&, node*, int);
	node * del(node*);
	friend std::ostream& operator<< (std::ostream&, BinaryTree&);
};
#endif // !BINTREE_H