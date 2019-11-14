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
	size_t Size() const;
	const T& operator[](const size_t index) const;
	T& operator[](const size_t index);
	Array<T>& operator=(const Array<T> &other);
	Array<T>& operator=(Array<T> &&other);
	T* begin();
	T* end();
	T* Find(const T &obj);
	T* Remove(T* it);
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
size_t Array<T>::Size() const {
	return m_size;
}

template<typename T>
const T& Array<T>::operator[](const size_t index) const {
	return m_ptr[index];
}

template<typename T>
T& Array<T>::operator[](const size_t index) {
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
template<typename T>
bool operator<(const Array<T> &arr1, const Array<T> &arr2) {
	return *(arr1[0]) < *(arr2[0]);
}

template<typename T>
bool operator>(const Array<T> &arr1, const Array<T> &arr2) {
	return *(arr1[0]) > *(arr2[0]);
}

template<typename T>
bool operator==(const Array<T> &arr1, const Array<T> &arr2) {
	return *(arr1[0]) == *(arr2[0]);
}
template<typename T>
std::ostream& operator<<(std::ostream &os, const Array<T> &arr) {
	if (arr[0] == nullptr)
		return os;
	os << "{ ";
	auto size = arr.Size();
	for (decltype(size) i = 0; i < size; ++i) {
		if (arr[i] == nullptr)
			continue;
		os << *(arr[i]) << ' ';
	}
	os << '}';
	return os;
}
template<typename T>
T* Array<T>::Find(const T &obj) {
	for (size_t i = 0; i < m_size; ++i)
		if (m_ptr[i] == obj)
			return &m_ptr[i];
	return nullptr;
}

template<typename T>
T* Array<T>::Remove(T *it) {
	it->~T();
	T* end = this->end();
	while ((it+1) != end) {
		*it = *(it + 1);
		++it;
	}
	*it = T();
	--m_size;
	return it;
}
#endif // !ARRAY_H