#include <iostream>

#include "rhomb.hpp"
#include "btree.hpp"

#include <iostream>

int main()
{
	Tree t;

	char command;
	double d1, d2;
	node *found;

	while (1) {
		std::cout << "Enter a command please" << std::endl;

		std::cin >> command;
		if (std::cin.eof()) {
			break;
		}
		switch (command) {
		case 'p':
			t.Print();
			break;
		case 'a':
			std::cin >> d1;
			std::cin >> d2;
			t.Add(Rhomb(d1, d2));
			std::cout << "ADDED A RHOMB WITH DIAGONALS: d1 = " << d1 << ", d2 = " << d2 << '.' << std::endl;
			std::cout << "Enter <p> to see the result." << std::endl;
			break;
		case 'r':
			std::cin >> d1;
			std::cin >> d2;
			t.Remove(Rhomb(d1, d2));
			std::cout << "REMOVED A RHOMB WITH DIAGONALS: d1 = " << d1 << ", d2 = " << d2 << '.' << std::endl;
			std::cout << "Enter <p> to see the result." << std::endl;
			break;
		case 'h':
			std::cout << "Existing commands: a, p, r, h." << std::endl;
			std::cout << "\ta d1 d2 - to add new figure with d1 d2 parameters to tree." << std::endl;
			std::cout << "\tr d1 d2 - to remove figure with d1 d2 parameters from tree." << std::endl;
			std::cout << "\tp - to print current tree." << std::endl;
			std::cout << "\th - to see help." << std::endl;
			std::cout << "\tf d1 d2 - to find object with the same parameters." << std::endl;
			break;
		case 'f':
			std::cin >> d1;
			std::cin >> d2;
			found = t.Find(Rhomb(d1, d2));
			if (found != nullptr) {
				std::cout << "Found: " << found->data << std::endl;
			}
			else {
				std::cout << "Nothing found" << std::endl;
			}
			break;
		default:
			std::cout << "Existing commands: a, p, r." << std::endl;
			break;
		}
	}

    return 0;
}