#ifndef FIELD_H
#define FIELD_H

#include <stdlib.h>

#define CL_WATER '~'
#define CL_SHIP '#'
#define CL_DESTROYED 'X'
#define CL_MISS 'o'

#define OR_VERTICAL 0
#define OR_HORIZONTAL 1

char get_cell(char *b, int x, int y){
    return b[y * 10 + x];
}

void set_cell(char *b, int x, int y, char c){
    b[y * 10 + x] = c;
}

void fill_field(char *b, char c){
    for(int i = 0; i < 100; i++)
        b[i] = c;
}

int can_be_ship(char *field, int x, int y){
    for(int i = -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            int cur_x = x + i;
            int cur_y = y + j;

            if(cur_x >= 0 && cur_x < 10 && cur_y >= 0 && cur_y < 10 
                && get_cell(field, x + i, y + j) == CL_SHIP){
                    return 0;
            }
        }
    }
    return 1;
}

int can_place(char *field, int x, int y, int dir, int size){
    if(dir == OR_HORIZONTAL){
        for(int i = x; i < x + size; i++)
            if(i >= 10 || !can_be_ship(field, i, y))
                return 0;
    }
    else{
        for (int i = y; i < y + size; i++)
            if (i >= 10 || !can_be_ship(field, x, i)){
                return 0;
            }
    }
    return 1;
}

void place_ship(char *field, int size){
    int x = rand() % 10;
    int y = rand() % 10;
    int dir = rand() % 2;
    while(!can_place(field, x, y, dir, size)){
        x = rand() % 10;
        y = rand() % 10;
        dir = rand() % 2;
    }
    if(dir == OR_HORIZONTAL){
        for(int i = x; i < x + size; i++)
            set_cell(field, i, y, CL_SHIP);
    }
    else{
        for (int i = y; i < y + size; i++)
            set_cell(field, x, i, CL_SHIP);
    }
}

void generate_field(char *field){
    fill_field(field, CL_WATER);
    for(int i = 4; i > 0; i--){
        for(int j = 0; j < 5 - i; j++){
            place_ship(field, i);
        }
    }
}

int check_field(char *field){
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 10; j++){
            if(get_cell(field, j, i) == CL_SHIP)
                return 1;
        }
    }
    return 0;
}
#endif