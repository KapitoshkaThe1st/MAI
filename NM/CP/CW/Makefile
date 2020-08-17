CFLAGS=-O3 -march=native -std=c++17 -pedantic -Wall -Wextra
all: main1.cpp main2.cpp benchmark1.cpp benchmark2.cpp
	g++ $(CFLAGS) -o prog1 main1.cpp
	g++ $(CFLAGS) -o prog2 main2.cpp
	g++ $(CFLAGS) -o benchmark1 benchmark1.cpp
	g++ $(CFLAGS) -o benchmark2 benchmark2.cpp
debug: main.cpp
	g++ $(CFLAGS) -g -fsanitize=address -fsanitize=undefined -o prog main.cpp

prof:
	g++ -g3 $(CFLAGS) -o benchmark1 benchmark1.cpp
