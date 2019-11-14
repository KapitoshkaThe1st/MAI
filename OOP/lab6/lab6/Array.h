#ifndef ARRAY_H
#define ARRAY_H

template<typename T>
class Array {
public:
	Array();
	Array(size_t s);
	Array(const Array<T> &other);
	Array(Array<T> &&other);
	~Array();
	size_t Size();
	T& operator[](size_t index);
	Array<T>& operator=(const Array<T> &other);
	Array<T>& operator=(Array<T> &&other);
	T* begin();
	T* end();
private:
	T *m_ptr;
	size_t m_size;
};

template<typename T>
Array<T>::Array() : m_ptr(nullptr), m_size(0) {}

template<typename T>
Array<T>::Array(size_t s) : m_size(s) {
	m_ptr = static_cast<T*>(malloc(sizeof(T) * m_size));
	for (size_t i = 0; i < m_size; ++i)
		new (m_ptr + i) T();		// placement new
}

template<typename T>
Array<T>::Array(const Array<T> &other) {
	m_size = other.m_size;
	memcpy(m_ptr, other.m_ptr, sizeof(T) * m_size);
}

template<typename T>
Array<T>::Array(Array<T> &&other) {
	m_ptr = other.m_ptr;
	m_size = other.m_size;
	other.m_ptr = nullptr;
	other.m_size = 0;
}

template<typename T>
Array<T>::~Array() {
	if (m_ptr != nullptr) {
		for (size_t i = 0; i < m_size; ++i)
			m_ptr[i].~T();
		free(m_ptr);
	}
}

template<typename T>
size_t Array<T>::Size() {
	return m_size;
}

template<typename T>
T& Array<T>::operator[](size_t index) {
	return m_ptr[index];
}

template<typename T>
Array<T>& Array<T>::operator=(const Array<T> &other) {
	m_size = other.m_size;
	memcpy(m_ptr, other.m_ptr, sizeof(T) * m_size);
	return *this;
}

template<typename T>
Array<T>& Array<T>::operator=(Array<T> &&other) {
	m_ptr = other.m_ptr;
	m_size = other.m_size;
	other.m_ptr = nullptr;
	other.m_size = 0;
	return *this;
}

template<typename T>
T* Array<T>::begin() {
	return &m_ptr[0];
}

template<typename T>
T* Array<T>::end() {
	return m_ptr + m_size;
}
#endif // !ARRAY_H