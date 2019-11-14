TARGET=main
LIB_NAME=libvector.so
lib: $(LIB_NAME)
$(LIB_NAME): vector.c
	gcc -fPIC -c vector.c
	gcc -shared -o $(LIB_NAME) vector.o
	sudo cp $(LIB_NAME) /usr/lib
init_time: $(LIB_NAME) init_time_main.c
	gcc -c init_time_main.c
	gcc init_time_main.o -L. -lvector -o it_$(TARGET)
full_dynamic: $(LIB_NAME) full_dynamic_main.c
	gcc -c full_dynamic_main.c
	gcc full_dynamic_main.o -ldl -o fd_$(TARGET)
clean:
	rm *.o
	sudo rm /usr/lib/$(LIB_NAME)
