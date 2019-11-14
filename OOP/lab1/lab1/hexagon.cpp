#include "hexagon.hpp"
#include <iostream>
#include <cmath>

Hexagon::Hexagon() : Hexagon(0.0) {

}

Hexagon::Hexagon(double s) : m_s(s) {
}

Hexagon::Hexagon(std::istream &is) {
	is >> m_s;
}

Hexagon::Hexagon(const Hexagon &orig) {
	m_s = orig.m_s;
}

void Hexagon::Print() {
	std::cout << "Regular hexagon { " << m_s << " }" << std::endl;
}

double Hexagon::Square() {
	const double pi = 3.14159265;
	return 6.0 * m_s*m_s*sin(2.0 * pi * 60.0 / 360.0)/2.0;
}

Hexagon::~Hexagon() {

}