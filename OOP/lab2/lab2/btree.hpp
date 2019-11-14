#ifndef BTREE_H
#define BTREE_H

#include "rhomb.hpp"
#include <iostream>

struct node {
	Rhomb data;
	node *left, *right, *parent;
};

class Tree {
private:
	node *root;

	node* min(node *tree);
	node* max(node *tree);
	void print(std::ostream &os, int tab, node *tree) const;
	void remove(node* tree);
public:
	Tree();
	~Tree();
	void Add(Rhomb val);
	void Remove(Rhomb val);
	node* Find(Rhomb val);
	Rhomb Min();
	Rhomb Max();
	void Print();
	friend std::ostream& operator<<(std::ostream &os, const Tree& tree);
};

#endif