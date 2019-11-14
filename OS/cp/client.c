#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "database.h"
#include "interact.h"
#include "field.h"
#include "render.h"

#define READ_BUFF_L 256

int menu(){
    printf("1: play\n");
    printf("2: rating\n");
    printf("3: exit\n");
    int mode;
    scanf("%d", &mode);
    return mode;
}

void login(int socket, char *nickname){
    msg_t msg;
    init_data_msg(&msg, nickname, strlen(nickname) + 1, MT_LOGIN, 0);
    send_msg(socket, &msg);
    close_msg(&msg);
}

record* get_rating(int socket, int *count){
    msg_t msg;
    init_data_msg(&msg, &msg, 1, MT_RATING, 0);
    send_msg(socket, &msg);
    close_msg(&msg);
    recieve_msg(socket, &msg);
    record *rec = (record*)malloc(msg.size);
    memcpy(rec, msg.buffer, msg.size);
    *count = msg.size / sizeof(record);
    close_msg(&msg);
    return rec;
}

void print_field(char *field){
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            printf("%c ", field[i * 10 + j]);
        }
        printf("\n");
    }
}

void render_ini(buffer *screen){
    int status = init_buffer(screen, 40, 15);
    assert(status > 0);

    char coord[10];
    for (int i = 0; i < 10; i++)
        coord[i] = '0' + i;

    fill_buffer(screen, ' ');

    draw_to_buffer(coord, 10, 1, screen, 2, 1);
    draw_to_buffer(coord, 1, 10, screen, 1, 2);

    draw_to_buffer(coord, 10, 1, screen, 15, 1);
    draw_to_buffer(coord, 1, 10, screen, 14, 2);

    set_background_color(BG_WHITE);
    set_text_color(CH_BLACK);
}

void restore_console(){
    set_background_color(BG_BLACK);
    set_text_color(CH_WHITE);
    printf("\033[2J");
    printf("\033[0;0f");
}

void set_console(){
    set_background_color(BG_WHITE);
    set_text_color(CH_BLACK);
    printf("\033[2J");
    printf("\033[0;0f");
}

int main(int argc, char **argv){
    set_console();
    if(argc < 3){
        printf("usage: client <port>\n");
        exit(EXIT_SUCCESS);
    }

    int port = atoi(argv[1]);

    int sock;
    if((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    if(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) < 0){
        perror("address convertion failed");
        exit(EXIT_FAILURE);
    }
    addr.sin_port = htons(port);

    if(connect(sock, (struct sockaddr*)&addr, sizeof(struct  sockaddr)) < 0){
        perror("connect failed");
        exit(EXIT_FAILURE);
    }

    char *nickname = argv[2]; 
    
    database base;

    int should_close = 0;
    int playing = 0;

    buffer screen;

    int status = init_buffer(&screen, 40, 15);
    assert(status > 0);

    render_ini(&screen);

    while (!should_close){

        int mode = 0;

        while(mode == 0){
            do{
                mode = menu();
            } while(mode < 1 || mode > 3);

            if(mode == 1){
                login(sock, nickname);
                playing = 1;
                printf("waiting for opponent...\n");
            }
            else if(mode == 2){
                int count = 0;
                record *recs = get_rating(sock, &count);
                printf("nickname:\twinrate:\twin:\tlose:\tleave:\n");
                for(int i = 0; i < count; i++){
                    int denum = recs[i].wins + recs[i].losses;
                    float winrate = denum == 0 ? 0 : (float)recs[i].wins / denum;
                    printf("%s\t\t%2.2f\t\t%d\t%d\t%d\n", recs[i].uname, winrate, 
                        recs[i].wins, recs[i].losses, recs[i].lefts);
                }

                mode = 0;
            }
            else if(mode == 3){
                restore_console();
                exit(EXIT_SUCCESS);
            }
        }

        msg_t rep;
        init_msg(&rep);

        do{
            close_msg(&rep);
            recieve_msg(sock, &rep);
        } while (rep.type != MT_FIELD);

        int sess_id = rep.id;

        printf("session id: %d\n", sess_id);

        char field[100];
        memcpy(field, rep.buffer, 100);
        char op_field[100];
        fill_field(op_field, CL_WATER);

        do{
            close_msg(&rep);
            recieve_msg(sock, &rep);
        } while (rep.type != MT_TURN);

        int turn = *(int *)rep.buffer;

        while(playing){
            draw_to_buffer(field, 10, 10, &screen, 2, 2);
            draw_to_buffer(op_field, 10, 10, &screen, 15, 2);
            render_r(&screen);

            if(turn){
                while(1){
                    shot sh;
                    do{
                        printf("enter shot coords\n");
                        scanf("%d", &sh.x);
                        scanf("%d", &sh.y);
                    }while(get_cell(op_field, sh.x, sh.y) != CL_WATER);

                    msg_t msg;
                    init_data_msg(&msg, &sh, sizeof(shot), MT_SHOT, sess_id);
                    send_msg(sock, &msg);
                    close_msg(&msg);

                    close_msg(&rep);
                    recieve_msg(sock, &rep);

                    if (rep.type == MT_ERROR){
                        printf("server crashed\n");
                        playing = 0;
                        should_close = 1;
                        break;
                    }

                    if (rep.type == MT_WIN){
                        set_cell(op_field, sh.x, sh.y, CL_DESTROYED);
                        printf("Congratulation! You won!\n");
                        mode = 0;
                        playing = 0;
                        break;
                    }
                    if (rep.type == MT_MISS){
                        printf("miss\n");
                        set_cell(op_field, sh.x, sh.y, CL_MISS);
                        turn = 0;
                        break;
                    }
                    printf("hit\n");
                    set_cell(op_field, sh.x, sh.y, CL_DESTROYED);
                    draw_to_buffer(field, 10, 10, &screen, 2, 2);
                    draw_to_buffer(op_field, 10, 10, &screen, 15, 2);
                    render_r(&screen);
                }
            }
            else{
                while(1){
                    msg_t msg;

                    recieve_msg(sock, &rep);

                    if(rep.type == MT_ERROR){
                        printf("server crashed\n");
                        playing = 0;
                        should_close = 1;
                        break;
                    }

                    shot *sh = (shot*)rep.buffer;
                    if (rep.type == MT_WIN){
                        printf("Congratulation! You won!\n");
                        mode = 0;
                        playing = 0;
                        break;
                    }
                    if (rep.type == MT_LOSE){
                        set_cell(field, sh->x, sh->y, CL_DESTROYED);
                        printf("You lose! Better luck next time!\n");
                        mode = 0;
                        playing = 0;
                        break;
                    }
                    if (rep.type == MT_MISS){
                        printf("miss\n");

                        set_cell(field, sh->x, sh->y, CL_MISS);
                        turn = 1;
                        break;
                    }
                    printf("hit\n");

                    set_cell(field, sh->x, sh->y, CL_DESTROYED);
                    draw_to_buffer(field, 10, 10, &screen, 2, 2);
                    draw_to_buffer(op_field, 10, 10, &screen, 15, 2);
                    render_r(&screen);
                }
            }
        }
    }

    restore_console();
}