#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <fstream>
#include <cstring>

const int MAX_BUFFER_SIZE = 2147483647;

int bufferSize = MAX_BUFFER_SIZE;
char *buffer;

void StringInit(int buffSize = bufferSize){
	bufferSize = buffSize;
	buffer = new char[buffSize];
}

void StringOff(){
	delete[] buffer;
}

class TString
{
private:
	char *str;
	int length;
public:
	TString() : str(nullptr), length(0) {}
	
	TString(const char ch){
		str = new char[2];
		str[0] = ch;
		str[1] = '\0';
		length = 1;
	}

	TString(TString &&orig){
		length = orig.length;
		str = orig.str;
		orig.str = nullptr;
		orig.length = 0;
	}

	TString(const TString &orig){
		str = new char[orig.length + 1];
		length = orig.length;
		memcpy(str, orig.str, (length + 1) * sizeof(char));
	}

	TString(const char *cstr){
		length = strlen(cstr);
		str = new char[length + 1];
		memcpy(str, cstr, (length + 1) * sizeof(char));
	}

	TString(char *cstr){
		length = strlen(cstr);
		str = new char[length + 1];
		memcpy(str, cstr, (length + 1) * sizeof(char));
	}

	int Length(){
		return length;
	}

	char At(int index){
		if(index >= 0 && index <= length)
			return str[index];
		return -1;
	}

	int ToInt(){
		return atoi(str);
	}
	unsigned long long ToULongLong(){
		return strtoull(str, 0, 0);
	}

	TString& ToUpper(){
		for(int i = 0; i < length; ++i){
			str[i] = toupper(str[i]);
		}
		return *this;
	}
	TString& ToLower(){
		for(int i = 0; i < length; ++i){
			str[i] = toupper(str[i]);
		}
		return *this;
	}

	operator char*() const {
		return str;
	}

	char* C_str() const {
		return str;
	}
	
	char& operator[](int index){
		return str[index];
	}

	TString& operator=(const TString &s){
		if(str != nullptr)
			delete[] str;
		length = s.length;
		str = new char[length + 1];
		memcpy(str, s.str, (length + 1) * sizeof(char));
		return *this;
	}
	TString& operator=(TString &&s){
		if(str != nullptr)
			delete[] str;
		length = s.length;
		str = s.str;
		s.str = nullptr;
		s.length = 0;
		return *this;
	}

	TString& operator+=(const TString &s){
		if(s.str == nullptr)
			return *this;
		char *temp = new char[length + s.length + 1];
		memcpy(temp, str, length * sizeof(char));
		memcpy(temp + length, s.str, (s.length + 1) * sizeof(char)); 
		length += s.length;
		if(str != nullptr)
			delete[] str;
		str = temp;
		return *this;
	}

	~TString(){
		delete[] str;
	}
	friend TString operator+(const TString&, const TString&);
	friend std::ostream& operator<<(std::ostream&, const TString&);
	friend std::istream& operator>>(std::istream&, TString&);
	friend std::ofstream& operator<<(std::ofstream&, const TString&);
	friend std::ifstream& operator>>(std::ifstream&, TString&);
	friend bool operator>(const TString&, const TString&);
	friend bool operator<(const TString&, const TString&);
	friend bool operator==(const TString&, const TString&);
	friend bool operator>=(const TString&, const TString&);
	friend bool operator<=(const TString&, const TString&);
	friend bool operator>(const TString &str1, const char *str2);
	friend bool operator<(const TString &str1, const char *str2);
	friend bool operator<=(const TString &str1, const char *str2);
	friend bool operator>=(const TString &str1, const char *str2);
	friend bool operator==(const TString &str1, const char *str2);
	friend bool operator>(const char *str1, const TString &str2);
	friend bool operator<(const char *str1, const TString &str2);
	friend bool operator<=(const char *str1, const TString &str2);
	friend bool operator>=(const char *str1, const TString &str2);
	friend bool operator==(const char *str1, const TString &str2);
};

bool operator>(const TString &str1, const TString &str2){
	return strcmp(str1.str, str2.str) > 0;
}
bool operator<(const TString &str1, const TString &str2){
	return strcmp(str1.str, str2.str) < 0;
}

bool operator<=(const TString &str1, const TString &str2){
	return !(strcmp(str1.str, str2.str) > 0);
}

bool operator>=(const TString &str1, const TString &str2){
	return !(strcmp(str1.str, str2.str) < 0);
}

bool operator==(const TString &str1, const TString &str2){
	return strcmp(str1.str, str2.str) == 0;
}

bool operator>(const TString &str1, const char *str2){
	return strcmp(str1.str, str2) > 0;
}
bool operator<(const TString &str1, const char *str2){
	return strcmp(str1.str, str2) < 0;
}

bool operator<=(const TString &str1, const char *str2){
	return !(strcmp(str1.str, str2) > 0);
}

bool operator>=(const TString &str1, const char *str2){
	return !(strcmp(str1.str, str2) < 0);
}

bool operator==(const TString &str1, const char *str2){
	return strcmp(str1.str, str2) == 0;
}

bool operator>(const char *str1, const TString &str2){
	return strcmp(str1, str2.str) > 0;
}
bool operator<(const char *str1, const TString &str2){
	return strcmp(str1, str2.str) < 0;
}

bool operator<=(const char *str1, const TString &str2){
	return !(strcmp(str1, str2.str) > 0);
}

bool operator>=(const char *str1, const TString &str2){
	return !(strcmp(str1, str2.str) < 0);
}

bool operator==(const char *str1, const TString &str2){
	return strcmp(str1, str2.str) == 0;
}

TString operator+(const TString &str1, const TString &str2){
	return std::move(TString(str1) += str2);
}

std::ostream& operator<<(std::ostream& os, const TString& s){
  	if(s.str == nullptr)
		return os;
    os << s.str 
    // << "(" << s.length << ")"
    ;  
    return os;  
}  

std::ofstream& operator<<(std::ofstream& op, const TString& s){
  	auto l = s.length + 1;
	op.write((char*)&l, (long)sizeof(l));
	op.write(s.C_str(), l * (long)sizeof(char));
    return op;  
}

std::ifstream& operator>>(std::ifstream& inp, TString& s){  
	if(s.str != nullptr)
		delete[] s.str;
	int l = 0;
	inp.read((char*)&l, sizeof(l));
	char *buff = new char[l];
	inp.read((char*)buff, sizeof(char) * l);
	s.str = buff;
	s.length = l - 1;
    return inp;  
}    

std::istream& operator>>(std::istream& is, TString& s){  
	if(s.str != nullptr)
		delete[] s.str;
	int len = 0;
	char ch;
	while((ch = is.get()) != ' ' && ch != '\t' && ch != '\n' && !is.eof()){
		if(len == bufferSize){
			is.ignore(MAX_BUFFER_SIZE, '\n');
			// is.ignore(MAX_BUFFER_SIZE, '\t');
			// is.ignore(MAX_BUFFER_SIZE, ' ');
			break;
		}
		buffer[len] = ch;
		++len;
	}
	s.str = new char[len+1];
	memcpy(s.str, buffer, len * sizeof(char));
	s.str[len] = '\0';
	s.length = len;
    return is;  
}  

#endif