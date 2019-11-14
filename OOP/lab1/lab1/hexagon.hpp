#ifndef HEXAGON_H
#define HEXAGON_H

#include <iostream>
#include "figure.hpp"

/* Правильный шестиугольник, заданный 1-й стороной */

class Hexagon : public Figure {
private:
	double m_s;
public:
	Hexagon();
	Hexagon(double);
	Hexagon(std::istream&);
	Hexagon(const Hexagon&);

	void Print() override;
	double Square() override;

	virtual ~Hexagon();
};

#endif