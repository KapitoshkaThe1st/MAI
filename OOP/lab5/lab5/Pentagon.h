#ifndef PENTAGON_H
#define PENTAGON_H

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>  

#include "Figure.h"

class Pentagon : public Figure {
private:
	double m_side;
	void print(std::ostream&) override;
public:
	Pentagon(std::istream&);
	Pentagon(double);
	Pentagon(Pentagon&);
	double Square();
	virtual ~Pentagon();

	Pentagon& operator=(const Pentagon&);
	friend std::istream& operator>>(std::istream&, Pentagon&);
};
#endif // !PENTAGON_H
