#include "Rhomb.h"
Rhomb::Rhomb(std::istream &is) {
	std::cin >> m_diag1 >> m_diag2;
}
Rhomb::Rhomb(double d1, double d2) : m_diag1(d1), m_diag2(d2) { }
Rhomb::Rhomb(Rhomb &orig) {
	m_diag1 = orig.m_diag1;
	m_diag2 = orig.m_diag2;
}
void Rhomb::Print(std::ostream& os = std::cout){
	std::cout << "rhomb(" << m_diag1 << ", " << m_diag2 << ")" << std::endl;
}
double Rhomb::Square() {
	return m_diag1 * m_diag2 / 2.0;
}
Rhomb::~Rhomb() {}

Rhomb& Rhomb::operator=(const Rhomb &other) {
	if (this == &other)
		return *this;
	m_diag1 = other.m_diag1;
	m_diag2 = other.m_diag2;
	return *this;
}
std::istream& operator>>(std::istream &is, Rhomb &r) {
	is >> r.m_diag1 >> r.m_diag2;
	return is;
}
std::ostream& operator<<(std::ostream &os, const Rhomb &r) {
os << "rhomb(" << r.m_diag1 << ", " << r.m_diag2 << ")" << std::endl;
return os;
}