#ifndef VECTOR_H
#define VECTOR_H

#include <cstring>
#include <iostream>

class TRecord;

template<typename T>
class TVector
{
private:
    T *data;
    size_t capacity;
    size_t size;
public:
    TVector(const size_t reserve = 1);
    ~TVector();
    T& operator[](const size_t);
    void Resize(const size_t);
    void PushBack(const T&);
    void PushFront(const T&);
    T& PopBack();
    T& PopFront();
    size_t Size();
    size_t Capacity();
    T& At(const size_t);
    int Empty();
    T* Begin();
    T* End();
    //friend void RadixSort(TVector<TRecord>&, int);
};

template<typename T>
TVector<T>::TVector(const size_t reserve) : capacity(reserve), size(0)
{
    data = new T[capacity];
}

template<typename T>
TVector<T>::~TVector() {
    delete[] data;
}

template<typename T>
T& TVector<T>::operator[](const size_t index) {
    return data[index];
}

template<typename T>
void TVector<T>::Resize(const size_t newCap){
    T *temp = new T[newCap];
    if(size < newCap)
        memcpy(temp, data, size * sizeof(T));
    else{
        memcpy(temp, data, newCap * sizeof(T));
        size = newCap;
    }
    delete [] data;
    capacity = newCap;
    data = temp;
}

template<typename T>
void TVector<T>::PushBack(const T& element){
    if(size >= capacity){
        Resize(2*size);
    }
    data[size] = element;   
    ++size;
}

template<typename T>
void TVector<T>::PushFront(const T& element){
    if(size >= capacity){
        Resize(2*size);
    }
    memcpy(data + 1, data, size * sizeof(T));
    data[0] = element;
    ++size;
}

template<typename T>
T& TVector<T>::PopBack(){
    if(size * 4 < capacity){
        Resize(capacity/2);
    }
    --size;
    return data[size];
}

template<typename T>
T& TVector<T>::PopFront(){
    if(size * 4 < capacity){
        Resize(capacity/2);
    }
    T temp = data[0];
    --size;
    memcpy(data, data + 1, size * sizeof(T));
    return temp;
}

template<typename T>
size_t TVector<T>::Size(){
    return size;
}

template<typename T>
size_t TVector<T>::Capacity(){
    return capacity;
}

template<typename T>
T& TVector<T>::At(const size_t index){
    if(index >= 0 && index <= size-1)
        return data[index];
    else{
        std::cerr << "attempt to go out of the array" << std::endl;
        exit(1);
    } 
} 

template<typename T>
int TVector<T>::Empty(){
    return size == 0;
}
template<typename T>
T* TVector<T>::Begin(){
    return data;
}
template<typename T>
T* TVector<T>::End(){
    return data + size - 1;
}

#endif