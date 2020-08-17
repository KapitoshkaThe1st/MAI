CFLAGS=-O3 -march=native -std=c++17 -pedantic -Wall -Wextra
all: prog2.cpp
	g++ $(CFLAGS) -o prog2 prog2.cpp
debug: prog2.cpp
	g++ $(CFLAGS) -g -fsanitize=address -fsanitize=undefined -o prog2 prog2.cpp

# prof:
# 	g++ -g3 $(CFLAGS) -o benchmark1 benchmark1.cpp
