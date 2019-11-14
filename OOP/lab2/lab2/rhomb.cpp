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

Rhomb& Rhomb::operator=(Rhomb &orig) {
	this->m_d1 = orig.m_d1;
	this->m_d2 = orig.m_d2;
	return *this;
}

bool Rhomb::operator>(Rhomb &r) {
	return true ? this->Square() > r.Square() : false;
}

bool Rhomb::operator<(Rhomb &r) {
	return true ? this->Square() < r.Square() : false;
}
bool Rhomb::operator<=(Rhomb &r) {
	return *this < r || *this == r;
}
bool Rhomb::operator>=(Rhomb &r) {
	return *this > r || *this == r;
}

bool Rhomb::operator==(Rhomb & r)
{
	return this->m_d1 == r.m_d1 && this->m_d2 == r.m_d2;
}

std::istream& operator>>(std::istream &is, Rhomb& r) {
	is >> r.m_d1 >> r.m_d2;
	return is;
}

std::ostream& operator<<(std::ostream &os, const Rhomb &r){
	os << "Rhomb { " << r.m_d1 << ", " << r.m_d2 << " }";
	return os;
}