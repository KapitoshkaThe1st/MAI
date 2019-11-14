#include <iostream>
#include <chrono>
#include <vector>
#include <time.h>
#include <cstring>
#include <algorithm> 

#include "radix.hpp"
#include "record.hpp"

const int LETTER_RANGE = 26;
const int NUM_RANGE = 10;
const char FST_CAP_LETTER_CODE = 'A';
const char FST_NUM_CODE = '0';

char* GenKey(){
	const int keyLength = 7;
	char *key = new char[keyLength];

	const char fstCapLetter = 'A';

	for(int i = 0; i < keyLength - 1; ++i){
		if(i >= 1 && i <= 3){
			key[i] = FST_NUM_CODE + static_cast<char>(rand() % NUM_RANGE);
		}
		else{
			key[i] = FST_CAP_LETTER_CODE + static_cast<char>(rand() % LETTER_RANGE);
		}
	}
	key[6] = '\0';

	return key;
}
char* GenValue(){
	const int maxValLength = 20;

	int valueLength = rand() % maxValLength + 1;

	char *value = new char[valueLength];

	for(int i = 0; i < valueLength - 1; ++i){

		char temp = FST_CAP_LETTER_CODE + static_cast<char>(rand() % LETTER_RANGE);

		if(temp == '\n' || temp == '\0' || temp == '\t')
		{
			temp = FST_CAP_LETTER_CODE + static_cast<char>(rand() % LETTER_RANGE);
		}

		value[i] = temp;
	}
	value[valueLength-1] = '\0';

	return value;
}

int MonotonyCheck(std::vector<TRecord> &v, int size){
	for(int i = 0; i < size - 1; ++i){
		if(strcmp(v[i].key, v[i+1].key) > 0){
			std::cout << v[i].key << std::endl;
			std::cout << v[i+1].key << std::endl;
			return 0;
		}
	}
	return 1;
}
int MonotonyCheck(TVector<TRecord> &v, int size){
	for(int i = 0; i < size - 1; ++i){
		//int result = strcmp(v[i].key, v[i+1].key)
		if(strcmp(v[i].key, v[i+1].key) > 0){
			std::cout << v[i].key << std::endl;
			std::cout << v[i+1].key << std::endl;
			return 0;
		}
	}
	return 1;
}

int funccmp(const void *val1, const void *val2 ){
    // std::cout << "here!" << std::endl;

	char *key1 = ((TRecord*)val1)->key;
	char *key2 = ((TRecord*)val2)->key;s

	if (strcmp(key1, key2) <= 0){
		return 0;
	}
	return 1;
}

int main(){
	const int count = 10000000;

	srand(time(0));
	rand();

	TVector<TRecord> custVec(10);
	std::vector<TRecord> vec;

	for(int i = 0; i < count; ++i) {
		TRecord temp;

		temp.key = GenKey();
		temp.value = GenValue();

		//temp.Print();

		custVec.PushBack(temp);
		vec.push_back(temp);
		//vec[i].Print();
	}

	int fstTime, scdTime;

	std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    RadixSort(custVec.Begin(), custVec.Size());

    end = std::chrono::system_clock::now();
 
    fstTime = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    start = std::chrono::system_clock::now();

	//std::sort(vec.begin(), vec.end(), funccmp);
    std::qsort(vec.data(), vec.size(), sizeof(TRecord), funccmp);

	end = std::chrono::system_clock::now();

	scdTime = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

	if(!MonotonyCheck(custVec, custVec.Size())){
		std::cout << "Radix sort made some mistakes!" << std::endl;
		return 0;
	}

	if(!MonotonyCheck(vec, vec.size())){
		std::cout << "Standart sort made some mistakes!" << std::endl;
		return 0;
	}

	std::cout << "Radix sort time: " << fstTime << "ms." << std::endl;
	std::cout << "Standart sort time: " << scdTime << "ms." << std::endl; 

	for(int i = 0; i < count; ++i){
		delete[] vec[i].key;
		delete[] vec[i].value;
	}
}
