TARGET=lab1
KEYS=-Wall -Wextra -Werror -pedantic -Wno-sign-compare -Wno-long-long -g3 -std=c++17 
DEBUG=-g
#-O0
all: main.o
	g++ main.o -o $(TARGET)
main.o: main.cpp vector.hpp queue.hpp
	g++ -c main.cpp -o main.o $(FLAGS)
debug: main_dbg.o
	g++ main_dbg.o -o $(TARGET)_dbg
main_dbg.o: main.cpp vector.hpp queue.hpp
	g++ -c main.cpp -o main_dbg.o $(DEBUG)
clean: 
	rm -f *.o $(TARGET)
