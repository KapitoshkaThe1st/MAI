#ifndef PENTAGON_H
#define PENTAGON_H

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>  

#include "Figure.h"
#include "Allocator.h"

class Pentagon : public Figure {
private:
	double m_side;
	void print(std::ostream&) override;
public:
	Pentagon(std::istream&);
	Pentagon(double);
	Pentagon(Pentagon&);
	double Square() const;
	virtual ~Pentagon();

	//static void* operator new(size_t, Allocator&);
	//static void operator delete(void*, Allocator&);
	Pentagon& operator=(const Pentagon&);
	friend std::istream& operator>>(std::istream&, Pentagon&);
};
#endif // !PENTAGON_H
