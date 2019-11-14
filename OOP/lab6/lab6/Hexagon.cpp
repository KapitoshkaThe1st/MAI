#include "Hexagon.h"

Hexagon::Hexagon(std::istream &is) {
	std::cin >> m_side;
}
Hexagon::Hexagon(double s) : m_side(s) { }
Hexagon::Hexagon(Hexagon &orig) {
	m_side = orig.m_side;
}
void Hexagon::print(std::ostream& os = std::cout){
	std::cout << "hexagon(" << m_side << ")";
}
double Hexagon::Square() {
	return 6.0 * m_side * m_side / 4.0 / tan(M_PI / 6.0);
}
Hexagon::~Hexagon() {}

Hexagon& Hexagon::operator=(const Hexagon &other) {
	if (this == &other)
		return *this;
	m_side = other.m_side;
	return *this;
}

std::istream& operator>>(std::istream &is, Hexagon &r) {
	is >> r.m_side;
	return is;
};

//void* Hexagon::operator new(size_t size, Allocator &allocator) {
//	return allocator.allocate();
//}
//
//void Hexagon::operator delete(void* to_deallocate, Allocator &allocator) {
//	allocator.deallocate(to_deallocate);
//}