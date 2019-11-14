#ifndef RENDER_H
#define RENDER_H

#include <stdlib.h> 
#include <stdio.h>
#include <assert.h>
#include <ctype.h>

#include "field.h"

typedef struct{
    int width;
    int height;
    char **data;
} buffer;

#define CH_BLACK "\033[30m"
#define CH_RED "\033[31m"
#define CH_GREEN "\033[32m"
#define CH_BROWN "\033[33m"
#define CH_BLUE "\033[34m"
#define CH_MAGENTA "\033[35m"
#define CH_CYAN "\033[36m"
#define CH_WHITE "\033[37m"

#define BG_BLACK "\033[40m"
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_BROWN "\033[43m"
#define BG_BLUE "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN "\033[46m"
#define BG_WHITE "\033[47m"

void set_background_color(const char *c){
    printf("%s", c);
}

void set_text_color(const char *c){
    printf("%s", c);
}

int init_buffer(buffer *b, int w, int h)
{
    b->width = w;
    b->height = h;
    if((b->data = (char**)malloc(h * sizeof(char*))) == NULL){
        return -1;
    }
    for(int i = 0; i < h; i++){
        if((b->data[i] = (char*)malloc((w + 1) * sizeof(char)))== NULL){
            return -1;
        }
        b->data[i][w] = '\0';
    }
    return 1;
}

void draw_to_buffer(char *f, int w, int h, buffer *b, int x, int y){
    int cur_x, cur_y;
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            cur_x = j + x;
            cur_y = i + y;
            if(cur_y >= 0 && cur_y < b->height && cur_x >= 0 && cur_x < b->width){
                b->data[cur_y][cur_x] = *(f + i * w + j);
            }
        }
    }
}

void fill_buffer(buffer *b, char c){
    for(int i = 0; i < b->height; i++)
        for(int j = 0; j < b->width; j++)
            b->data[i][j] = c;
}

void render_r(buffer *b){
    printf("\e[1;1H\e[2J");
    printf("\033[0;0f");
    for(int i = 0; i < b->height; i++){
        for(int j = 0; j < b->width; j++){
            char c = b->data[i][j];
            if(c == ' ' || (isalpha(c) && c != 'X' && c != 'o') || isdigit(c)){
                set_text_color(CH_BLACK);
                printf("%c ", c);
            }
            else{
                switch (c)
                {
                    case CL_SHIP:
                        set_text_color(CH_BROWN);
                        break;
                    case CL_WATER:
                        set_text_color(CH_BLUE);
                        break;
                    case CL_DESTROYED:
                        set_text_color(CH_RED);
                        break;
                    case CL_MISS:
                        set_text_color(CH_CYAN);
                        break;
                    default:
                        break;
                }
                printf("%c ", c);
            }  
        } 
        printf("\n");
    }
}

#endif