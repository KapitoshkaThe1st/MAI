#include <iostream>
#include <memory>

#include "BinaryTree.h"
#include "Figure.h"
#include "Rhomb.h"
#include "Pentagon.h"
#include "Hexagon.h"

using namespace std;

shared_ptr<Figure> CreateFigure(char type);

int main()
{
	BinaryTree<Figure> tree;

	char command = 0;
	char fig_type = 0;


	while (1) {
		cout << "Enter a command <a/f/r/p/h>" << endl;
		cin >> command;
		if (cin.eof())
			break;
		if (command == 'a') {
			cout << "Enter a figure <r/p/h> d1 (d2) to insert" << endl;
			cin >> fig_type;
			if (fig_type != 'h' || fig_type != 'p' || fig_type != 'r')
				continue;
			tree.Insert(CreateFigure(fig_type));
		}
		else if (command == 'f') {
			cout << "Enter a figure <r/p/h> d1 (d2) to find" << endl;
			cin >> fig_type;
			if (fig_type != 'h' || fig_type != 'p' || fig_type != 'r')
				continue;
			shared_ptr<Figure> temp = tree.Find(CreateFigure(fig_type));
			cout << "Found: " << *temp << endl;
		}
		else if (command == 'p') {
			cout << "Current state of the tree: " << endl;
			tree.Print();
		}
		else if (command == 'r') {
			cout << "Enter a figure <r/p/h> d1 (d2) to remove" << endl;
			cin >> fig_type;
			if (fig_type != 'h' || fig_type != 'p' || fig_type != 'r')
				continue;
			tree.Erase(CreateFigure(fig_type));
		}
		else if (command == 'h') {
			cout << 
R"(commands:
	a -- add,
	f -- find
	r -- remove,
	p -- print,
	h -- help.
figures:
	p l -- pentgon length of side,
	h l -- hexagon length of side,
	r d1 d2 -- rhomb by two diagonals d1, d2.)" << endl;
		}
	}

    return 0;
}

shared_ptr<Figure> CreateFigure(char type) {
	Figure *fig = nullptr;
	if (type == 'r')
		fig = new Rhomb(cin);
	else if (type == 'p')
		fig = new Pentagon(cin);
	else if (type == 'h')
		fig = new Hexagon(cin);
	return shared_ptr<Figure>(fig);
}