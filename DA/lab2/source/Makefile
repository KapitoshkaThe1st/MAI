TARGET=lab2
CFLAGS=-Wall -Wextra -Werror -pedantic -Wno-sign-compare -Wno-long-long -lm -O3 -std=c++17 
DEBUG=-ggdb -g3

all: main.o
	g++ main.o $(LDFLAGS) -o $(TARGET)
main.o: main.cpp
	g++ -c main.cpp -o main.o $(CFLAGS)
debug: main_dbg.o
	g++ main_dbg.o -o $(TARGET)_dbg
main_dbg.o: main.cpp
	g++ -c main.cpp -o main_dbg.o $(DEBUG)
clean: 
	rm -f *.o $(TARGET)
