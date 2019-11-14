#ifndef PIPELINE_H
#define PIPELINE_H

#include <functional>

#include "Array.h"

template<typename T>
class Pipeline {
	typedef Array<std::function<T>> actions;
private:
	size_t m_size;
	size_t m_count;
	actions m_actions;
public:
	Pipeline(size_t size) : m_size(size), m_count(0), m_actions(actions(size)) {}
	void Push(const std::function<T> &func) {
		m_actions[m_count] = func;
		++m_count;
	}
	void PerformAll() {
		for (size_t i = 0; i < m_count; ++i)
			m_actions[i]();
	}
	void Clear() {
		m_count = 0;
	}
};

#endif // !PIPELINE_H