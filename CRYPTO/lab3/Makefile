CFLAGS=-O3 -march=native -std=c++17 -pedantic -Wall -Wextra
all: hash_string hash_file main.cpp blake.h
	g++ -O3 -march=native -std=c++17 -pedantic -Wall -Wextra -o prog main.cpp
hash_string: hash_string.cpp blake.h
	g++ $(CFLAGS) -o hash_string hash_string.cpp
hash_file: hash_file.cpp blake.h
	g++ $(CFLAGS) -o hash_file hash_file.cpp