#include "Pentagon.h"
Pentagon::Pentagon(std::istream &is) {
	std::cin >> m_side;
}
Pentagon::Pentagon(double s) : m_side(s) { }
Pentagon::Pentagon(Pentagon &orig) {
	m_side = orig.m_side;
}
void Pentagon::print(std::ostream& os = std::cout) {
	std::cout << "pentagon(" << m_side << ")";
}
double Pentagon::Square() const {
	return 5.0 * m_side * m_side / 4.0 / tan(M_PI / 5.0);
}
Pentagon::~Pentagon() {}

Pentagon& Pentagon::operator=(const Pentagon &other) {
	if (this == &other)
		return *this;
	m_side = other.m_side;
	return *this;
}
std::istream& operator>>(std::istream &is, Pentagon &r) {
	is >> r.m_side;
	return is;
}

//void* Pentagon::operator new(size_t size, Allocator &allocator) {
//	std::cout << "custom allocator operator new" << std::endl;
//	return allocator.allocate();
//}
//
//void Pentagon::operator delete(void* to_deallocate, Allocator &allocator) {
//	std::cout << "custom allocator operator delete" << std::endl;
//	allocator.deallocate(to_deallocate);
//}