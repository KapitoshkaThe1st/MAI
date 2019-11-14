#ifndef RHOMB_H
#define RHOMB_H

#include <iostream>
#include "figure.hpp"

/* Ромб, заданный двумя диагоналями */

class Rhomb : public Figure{
private:
	double m_d1, m_d2;
public:
	Rhomb();
	virtual ~Rhomb();
	
	Rhomb(double, double);
	Rhomb(std::istream&);
	Rhomb(const Rhomb&);

	void Print() override;
	double Square() override;

	Rhomb& operator=(Rhomb&);
	bool operator>(Rhomb&);
	bool operator<(Rhomb&);
	bool operator<=(Rhomb&);
	bool operator>=(Rhomb&);
	bool operator==(Rhomb&);

	friend std::ostream& operator<<(std::ostream&, const Rhomb&);
	friend std::istream& operator>>(std::istream&, Rhomb&);
};

#endif