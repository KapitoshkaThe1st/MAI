CFLAGS=-O3 -march=native -std=c++17 -pedantic -Wall -Wextra
all: prog1.cpp
	g++ $(CFLAGS) -o prog1 prog1.cpp
debug: prog1.cpp
	g++ $(CFLAGS) -g -fsanitize=address -fsanitize=undefined -o prog1 prog1.cpp

# prof:
# 	g++ -g3 $(CFLAGS) -o benchmark1 benchmark1.cpp
