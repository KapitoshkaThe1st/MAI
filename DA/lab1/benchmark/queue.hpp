#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>

template <typename T>
class TQueue;

template <typename T>
class TPage{
private:
	size_t size;
	size_t tail;
	size_t head;
	TPage<T> *next;
	T *data;
public:
	TPage(size_t pageSize) : size(pageSize), tail(0), head(0), next(nullptr){
		data = new T[size];
	}
	~TPage(){

		if(data != nullptr){
			delete[] data;
		}
	}
	void Push(const T& element){
		data[tail] = element;
		++tail;
	}
	T& Pop(){
		++head;
		return data[head - 1];
	}
	T& First(){
		return data[head];
	}
	T& Last(){
		return data[tail-1];
	}
	int Long(){
		return tail == size;
	}
	int Short(){
		return head == size;
	}
	int Empty(){
		return head == tail;
	}

	template <typename TT>
	friend class TQueue;

};

template <typename T>
class TQueue{
private:
	TPage<T> *head;
	TPage<T> *tail;
	size_t size;
	size_t pageSize;
	TPage<T>* CreatePage(size_t size){
		TPage<T> *page = new TPage<T>(size);
		return page;
	}
public:
	TQueue(size_t pageS = 1000) : size(0), pageSize(pageS) {
		head = tail = nullptr;
	}
	~TQueue(){
		while(!Empty()){
			Pop();
		}
		if(head != nullptr){
			delete head;
		}
	}
	void Push(const T &element){
		if(tail == nullptr){
			head = tail = CreatePage(pageSize);
		}
		if(tail->Long()){
			tail->next = CreatePage(pageSize);
			tail = tail->next;
		}
		tail->Push(element);
		++size;

	}
	T& Pop(){
		if(head->Short()){

			TPage<T>* temp = head;
			head = head->next;
			delete temp;
		}
		--size;
		return head->Pop();
	}
	T& First(){
		return head->First();
	}
	T& Last(){
		return tail->Last();
	}
	int Empty(){
		return size == 0;
	}
	int Size(){
		return size;
	}
};

#endif
