#include "rhomb.hpp"
#include <iostream>

Rhomb::Rhomb() : Rhomb(0.0, 0.0) {

}

Rhomb::Rhomb(double d1, double d2) : m_d1(d1), m_d2(d2){

}

Rhomb::Rhomb(std::istream &is){
	is >> m_d1;
	is >> m_d2;
}

Rhomb::Rhomb(const Rhomb &orig){
	m_d1 = orig.m_d1;
	m_d2 = orig.m_d2;
}

void Rhomb::Print(){
	std::cout << "Rhomb { " << m_d1 << ", " << m_d2 << " }" << std::endl;
}

double Rhomb::Square(){
	return m_d1 * m_d2 / 2.0;
}

Rhomb::~Rhomb(){

}