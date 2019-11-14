#ifndef FIGURE_H
#define FIGURE_H

#include <iostream>

class Figure {
public:
	Figure() { };
	virtual double Square() = 0;
	virtual ~Figure() {}
private:
	virtual void print(std::ostream& os = std::cout) = 0;
	friend std::ostream& operator<<(std::ostream &, Figure&);
};

bool operator==(Figure&, Figure&);
bool operator<(Figure&, Figure&);
bool operator>(Figure&, Figure&);
bool operator<=(Figure&, Figure&);
bool operator>=(Figure&, Figure&);

#endif // !FIGURE_H