#ifndef CRITERIA_H
#define CRITERIA_H

#include "Figure.h"
#include "Rhomb.h"
#include "Pentagon.h"
#include "Hexagon.h"

template<typename T>
class Criteria {
public:
	Criteria(T *obj) {}
	virtual bool isIt(T *other) = 0;
};

template<typename T>
class All : public Criteria<T> {
public:
	All() = default;
	virtual bool isIt(T *other) override {
		return true;
	}
};

template<typename T, typename TT = Figure>
class Is : public Criteria<TT> {
public:
	Is(TT *obj) : m_obj(obj) {}
	virtual bool isIt(T *other) override {
		return (dynamic_cast<T*>(m_obj) == nullptr ? false : true);
	}
private:
	TT *m_obj;
};

template<typename T>
class BySquare : public Criteria<T> {
public:
	BySquare(T *obj) : m_obj(obj), m_size(size) {}
	virtual bool isIt(T *other) override {
		return m_obj->Square() == m_size;
	}
private:
	T *m_obj;
	double m_size;
};
#endif // !CRITERIA_H