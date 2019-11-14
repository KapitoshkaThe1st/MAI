#ifndef HEXAGON_H
#define HEXAGON_H

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>  

#include "Figure.h"

class Hexagon : public Figure {
private:
	double m_side;
	void print(std::ostream&) override;
public:
	Hexagon(std::istream&);
	Hexagon(double);
	Hexagon(Hexagon&);
	double Square();
	virtual ~Hexagon();

	Hexagon& operator=(const Hexagon&);
	friend std::istream& operator>>(std::istream&, Hexagon&);
};
#endif // !HEXAGON_H