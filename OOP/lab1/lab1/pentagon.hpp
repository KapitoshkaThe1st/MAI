#ifndef PENTAGON_H
#define PENTAGON_H

#include <iostream>
#include "figure.hpp"

/* Правильный пятиугольник, заданный 1-й стороной */

class Pentagon : public Figure {
private:
	double m_s;
public:
	Pentagon();
	Pentagon(double);
	Pentagon(std::istream&);
	Pentagon(const Pentagon&);

	void Print() override;
	double Square() override;

	virtual ~Pentagon();
};

#endif