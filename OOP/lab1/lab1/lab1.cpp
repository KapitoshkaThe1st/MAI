#include <iostream>

#include "figure.hpp"
#include "rhomb.hpp"
#include "pentagon.hpp"
#include "hexagon.hpp"

int main() {
	Figure *f;

	char fig;
	double a, b;

	while (1) {
		std::cin >> fig;
		if (std::cin.eof()) {
			break;
		}
		if (fig == 'r') {
			std::cin >> a >> b;
			f = new Rhomb(a, b);
			f->Print();
			std::cout << "Square: " << f->Square() << std::endl;
		}
		else if (fig == 'p') {
			std::cin >> a;
			f = new Pentagon(a);
			f->Print();
			std::cout << "Square: " << f->Square() << std::endl;
		}
		else {
			std::cin >> a;
			f = new Hexagon(a);
			f->Print();
			std::cout << "Square: " << f->Square() << std::endl;
		}
	}
	return 0;
}