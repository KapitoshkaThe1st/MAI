#include <iostream>
#include <memory>
#include <vector>

#include "Array.h"
#include "QuickSort.h"
#include "Allocator.h"

#include "Figure.h"
#include "Hexagon.h"
#include "Pentagon.h"
#include "Rhomb.h"

using namespace std;

const size_t page_size = 65536;
const size_t obj_size = max({ sizeof(Hexagon), sizeof(Pentagon) , sizeof(Rhomb) });

Allocator Figure::allocator = Allocator(page_size, obj_size);

int main()
{
	size_t size = 0;

	cout << "How much figures do you want to sort?" << endl;
	cin >> size;

	Array<shared_ptr<Figure>> array(size);

	auto comparator = [](auto &f1, auto &f2) -> int {
		return *f1 == *f2 ? 0 : *f1 < *f2 ? -1 : 1; };

	auto gen_figure = [](char type) -> shared_ptr<Figure> {
		switch (type) {
		case 'p':
			return shared_ptr<Figure>(new Pentagon(cin));
		case 'h':
			return shared_ptr<Figure>(new Hexagon(cin));
		case 'r':
			return shared_ptr<Figure>(new Rhomb(cin));
		default:
			return shared_ptr<Figure>(nullptr);
		}
	};

	char type;
	for (int i = 0; i < size; ++i) {
		cout << "Enter figure, please" << endl;
		cin >> type;
		array[i] = gen_figure(type);
	}

	cout << "You have entered these figures: " << endl;
	for (auto el : array)
		cout << *el << ' ';
	cout << endl;

	quick_sort(array, 0, size - 1, comparator);

	cout << "Now they are sorted by their squares: " << endl;
	for (auto el : array)
		cout << *el << ' ';
	cout << endl;

	system("pause");
    return 0;
}