#include "Figure.h"
std::ostream& operator<<(std::ostream &os, Figure &f) {
	dynamic_cast<Figure*>(&f)->print(os);
	//f.print(os);
	return os;
}
bool operator==(const Figure &f1, const Figure &f2) {
	return f1.Square() == f2.Square();
}
bool operator<(const Figure &f1, const Figure &f2) {
	return f1.Square() < f2.Square();
}
bool operator>(const Figure &f1, const Figure &f2) {
	return f1.Square() > f2.Square();
}
bool operator<=(const Figure &f1, const Figure &f2) {
	return f1.Square() <= f2.Square();
}
bool operator>=(const Figure &f1, const Figure &f2) {
	return f1.Square() >= f2.Square();
}

void* Figure::operator new(size_t size) {
	//std::cout << "custom allocator operator new" << std::endl;
	return allocator.allocate();
}

void Figure::operator delete(void* to_deallocate) {
	//std::cout << "custom allocator opearotr delete" << std::endl;
	allocator.deallocate(to_deallocate);
}