#include "pentagon.hpp"
#include <iostream>
#include <cmath>

Pentagon::Pentagon() : Pentagon(0.0) {

}

Pentagon::Pentagon(double s) : m_s(s) {

}

Pentagon::Pentagon(std::istream &is) {
	is >> m_s;
}

Pentagon::Pentagon(const Pentagon &orig) {
	m_s = orig.m_s;
}

void Pentagon::Print() {
	std::cout << "Regular pentagon { " << m_s << " }" << std::endl;
}

double Pentagon::Square(){
	const double pi = 3.14159265;
	return m_s*m_s/4.0*sqrt(25+10*sqrt(5));
}

Pentagon::~Pentagon() {

}