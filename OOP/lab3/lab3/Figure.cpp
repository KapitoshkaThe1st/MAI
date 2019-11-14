#include "Figure.h"
std::ostream& operator<<(std::ostream &os, Figure &f) {
	f.Print(os);
	return os;
}
bool operator==(Figure &f1, Figure &f2) {
	return f1.Square() == f2.Square();
}
bool operator<(Figure &f1, Figure &f2) {
	return f1.Square() < f2.Square();
}
bool operator>(Figure &f1, Figure &f2) {
	return f1.Square() > f2.Square();
}
bool operator<=(Figure &f1, Figure &f2) {
	return f1.Square() <= f2.Square();
}
bool operator>=(Figure &f1, Figure &f2) {
	return f1.Square() >= f2.Square();
}