#include <iostream>
#include <fstream>
#include <cstring>
#include "avl.hpp"
#include "custom_string.hpp"

int main()
{
	const int BUFFER_SIZE = 256;

	std::ios_base::sync_with_stdio(0);

	StringInit(BUFFER_SIZE);
	

	TAVL<TString, unsigned long long> tree;
	
	TString str1;
	TString str2;
	TString str3;


	while(1){
		std::cin >> std::ws;
		if (std::cin.eof()){
			break;
		}
		std::cin >> str1;
		if(str1 == "+"){
			std::cin >> str2;
			if(!isalpha(str2[0])){
				continue;
			}
			std::cin >> str3;
			if (!isdigit(str3[0])){
				continue;
			}
			if(tree.Insert(str2.ToUpper(), str3.ToULongLong())){
				std::cout << "OK" << std::endl;
			}
			else{
				std::cout << "Exist" << std::endl;
			}
		}
		else if(str1 == "-"){
			std::cin >> str2;
			if (!isalpha(str2[0])){
				continue;
			}
			if (tree.Remove(str2.ToUpper())){
				std::cout << "OK" << std::endl;
			}
			else{
				std::cout << "NoSuchWord" << std::endl;
			}
		}
		else if(str1 == "!"){
			std::cin >> str2;
			if(str2 == "Save"){
				std::cin >> str3;
				std::ofstream ofs(str3, std::ios::binary);
				if(ofs){
					tree.Serialize(ofs);
					std::cout << "OK" << std::endl;
				}
				else{
					std::cout << "ERROR: Couldn't create file" << std::endl;
				}
			}
			else if(str2 == "Load"){
				std::cin >> str3;
				std::ifstream ifs(str3, std::ios::binary);
				if (ifs)
				{
					TAVL<TString, unsigned long long> temp;
					if(!temp.Deserialize(ifs)){
						std::cout << "ERROR: Opened file isn't serialized dict" << std::endl;
						continue;
					}
					else{
						tree.Swap(temp);
						std::cout << "OK" << std::endl;
					}
				}
				else{
					std::cout << "ERROR: Couldn't open file" << std::endl;
				}
			}
		}
		else if(isalpha(str1[0])){
			unsigned long long *value;
			if((value = tree.Find(str1.ToUpper())) != nullptr){
				std::cout << "OK: " << *value << std::endl;
			}
			else{
				std::cout << "NoSuchWord" << std::endl;
			}
		}
	}

	StringOff();

	return 0;
}