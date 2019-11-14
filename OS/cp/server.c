#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

#include "vector.h"
#include "database.h"
#include "interact.h"
#include "field.h"

#define MAX_CLIENT_C 256
#define READ_BUFF_L 256

#define MAX_SESSIONS_C 16

#define QUEUE_CAP 16

typedef struct{
    int sock;
    record *rec;
} player;

typedef struct{
    player player1;
    player player2;
    char field1[100];
    char field2[100];
    int turn;
} session;

int should_close = 0;

void sigint_handler(int signal){
    should_close = 1;
}

void end_session(session *s, int winner){
    assert(s != NULL);
    msg_t w_msg;
    init_data_msg(&w_msg, &w_msg, 1, MT_WIN, 0);

    if(winner == 0){
        send_msg(s->player1.sock, &w_msg);
        s->player1.rec->wins++;
        s->player2.rec->losses++;
    }
    else{
        send_msg(s->player2.sock, &w_msg);
        s->player2.rec->wins++;
        s->player1.rec->losses++;
    }
    close_msg(&w_msg);
}


int main(int argc, char **argv){
    char read_buff[READ_BUFF_L];

    if(argc < 3){
        printf("usage: server <port> <database_file>");
        exit(EXIT_SUCCESS);
    }

    signal(SIGINT, sigint_handler);

    database base = load(argv[2]);

    vector queue;
    vector_init(&queue, QUEUE_CAP, player);

    session *sessions[MAX_SESSIONS_C];
    for(int i = 0; i < MAX_SESSIONS_C; i++)
        sessions[i] = NULL;

    srand(time(NULL));
    rand();
    int port = atoi(argv[1]);
    printf("port: %d\n", port);

    int client_socks[MAX_CLIENT_C];
    for(int i = 0; i < MAX_CLIENT_C; i++)
        client_socks[i] = 0;

    int master_sock;
    if((master_sock = socket(AF_INET, SOCK_STREAM, 0)) == 0){
        perror("master socket failed\n");
        exit(EXIT_FAILURE);
    }
    
    int opt = 1;
    if(setsockopt(master_sock, SOL_SOCKET, SO_REUSEADDR
        | SO_REUSEPORT, &opt, sizeof(opt)) < 0){
        perror("set socket options failed\n");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if(bind(master_sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in)) < 0){
        perror("master socket bind failed\n");
        exit(EXIT_FAILURE);
    }

    if(listen(master_sock, 3) < 0){
        perror("listen failed\n");
        exit(EXIT_FAILURE);
    }

    fd_set read_fd;

    while (!should_close){
        FD_ZERO(&read_fd);
        FD_SET(master_sock, &read_fd);

        int max_fd = master_sock;

        for(int i = 0; i < MAX_CLIENT_C; i++)
            if(client_socks[i] > 0){
                FD_SET(client_socks[i], &read_fd);
                if(max_fd < client_socks[i])
                    max_fd = client_socks[i];
            }

        if(select(max_fd + 1, &read_fd, NULL, NULL, NULL) < 0){
            if(errno != EINTR){
                perror("select failed\n");
                exit(EXIT_FAILURE);
            }
        }
        else{
            if (FD_ISSET(master_sock, &read_fd)){
                printf("new connection\n");
                int new_sock;
                if((new_sock = accept(master_sock, NULL, 0)) < 0){
                    perror("accept failed\n");
                    exit(EXIT_FAILURE);
                }
                printf("connected: %d\n", new_sock);
                for(int i = 0; i < MAX_CLIENT_C; i++){
                    if(client_socks[i] == 0){
                        client_socks[i] = new_sock;
                        break;
                    }
                }
            }
            for(int i = 0; i < MAX_CLIENT_C; i++){
                if(FD_ISSET(client_socks[i], &read_fd)){
                    printf("processing client: %d\n", client_socks[i]);
                    msg_t req;
                    if(recieve_msg(client_socks[i], &req) == 0){
                        printf("disconnected\n");

                        int queue_size = vector_size(&queue);
                        for(int j = 0; j < queue_size; j++){
                            player temp;
                            vector_fetch(&queue, j, &temp);
                            if(temp.sock == client_socks[i]){
                                printf("dequeued disconnected player\n");
                                vector_erase(&queue, j);
                                break;
                            }
                        }
                        for(int j = 0; j < MAX_SESSIONS_C; j++){
                            int found = 0;
                            if(sessions[j] != NULL){
                                if(sessions[j]->player1.sock == client_socks[i]){
                                    sessions[j]->player1.rec->lefts++;
                                    end_session(sessions[j], 1);
                                    found = 1;
                                }
                                if(sessions[j]->player2.sock == client_socks[i]){
                                    sessions[j]->player2.rec->lefts++;
                                    end_session(sessions[j], 0);
                                    found = 1;
                                }
                                if(found){
                                    printf("session ended because of disconnected player\n");
                                    free(sessions[j]);
                                    sessions[j] = NULL;
                                    break;
                                }
                            }
                            
                        }
                        printf("before closing\n");
                        close(client_socks[i]);
                        client_socks[i] = 0;
                    }
                    else{
                        switch (req.type){
                            case MT_LOGIN:{
                                    record *rec;
                                    printf("login %s\n", (char*)req.buffer);
                                    if((rec = find_rec(&base, req.buffer)) == NULL){
                                        printf("player %s not found\n", (char*)req.buffer);
                                        record new_rec;
                                        strcpy(new_rec.uname, req.buffer);
                                        new_rec.wins = 0;
                                        new_rec.losses = 0;
                                        new_rec.lefts = 0;

                                        add(&base, new_rec);
                                        printf("added to database\n");
                                        rec = get_rec(&base, base.count - 1);
                                    }
                                    printf("statistics:\n");
                                    printf("\twins: %d\n", rec->wins);
                                    printf("\tlosses: %d\n", rec->losses);
                                    printf("\tlefts: %d\n", rec->lefts);

                                    player p = {client_socks[i], rec};
                                    vector_push_front(&queue, &p);
                                    printf("queued\n");

                                    if(vector_size(&queue) >= 2){
                                        printf("start new session\n");
                                        session *s;
                                        int i;
                                        for(i = 0; i < MAX_SESSIONS_C; i++){
                                            if(sessions[i] == NULL){
                                                sessions[i] = malloc(sizeof(session));
                                                s = sessions[i];
                                                break;
                                            }
                                        }
                                        vector_back(&queue, &s->player1);
                                        vector_pop_back(&queue);
                                        vector_back(&queue, &s->player2);
                                        vector_pop_back(&queue);
                                        s->turn = 0;
                                        generate_field(s->field1);
                                        generate_field(s->field2);
                                        // fill_field(s->field1, CL_WATER);
                                        // set_cell(s->field1, 0, 0, CL_SHIP);
                                        // fill_field(s->field2, CL_WATER);
                                        // set_cell(s->field2, 0, 0, CL_SHIP);

                                        msg_t msg;
                                        init_data_msg(&msg, s->field1, 100, MT_FIELD, i);
                                        send_msg(s->player1.sock, &msg);
                                        close_msg(&msg);
                                        init_data_msg(&msg, s->field2, 100, MT_FIELD, i);
                                        send_msg(s->player2.sock, &msg);
                                        close_msg(&msg);

                                        int turn = 1;
                                        init_data_msg(&msg, &turn, sizeof(int), MT_TURN, 0);
                                        send_msg(s->player1.sock, &msg);
                                        close_msg(&msg);
                                        turn = 0;
                                        init_data_msg(&msg, &turn, sizeof(int), MT_TURN, 0);
                                        send_msg(s->player2.sock, &msg);
                                        close_msg(&msg);
                                    }
                                }
                                break;
                            case MT_SHOT:{
                                    printf("shot\n");

                                    shot *sh = (shot*)req.buffer;
                                    int id = req.id;
                                    printf("session id: %d\n", id);

                                    if(sessions[id]->player1.sock == client_socks[i] && sessions[id]->turn == 0){
                                        printf("player1 turn\n");
                                        msg_t msg;
                                        if(get_cell(sessions[id]->field2, sh->x, sh->y) == CL_SHIP){
                                            printf("hit: %d-%d\n", sh->x, sh->y);
                                            set_cell(sessions[id]->field2, sh->x, sh->y, CL_DESTROYED);
                                            if(!check_field(sessions[id]->field2)){
                                                init_data_msg(&msg, &id, 1, MT_WIN, id);
                                                send_msg(sessions[id]->player1.sock, &msg);
                                                req.type = MT_LOSE;
                                            }  
                                            else{
                                                init_data_msg(&msg, &id, 1, MT_HIT, id);
                                                send_msg(sessions[id]->player1.sock, &msg);
                                                req.type = MT_HIT;
                                            }
                                        }
                                        else{
                                            printf("miss: %d-%d\n", sh->x, sh->y);
                                            set_cell(sessions[id]->field2, sh->x, sh->y, CL_MISS);
                                            init_data_msg(&msg, &id, 1, MT_MISS, id);
                                            send_msg(sessions[id]->player1.sock, &msg);
                                            req.type = MT_MISS;
                                        }
                                        close_msg(&msg);

                                        send_msg(sessions[id]->player2.sock, &req);
                                    }
                                    else{
                                        printf("player2 turn\n");
                                        msg_t msg;
                                        if(get_cell(sessions[id]->field1, sh->x, sh->y) == CL_SHIP){
                                            printf("hit: %d-%d\n", sh->x, sh->y);
                                            set_cell(sessions[id]->field1, sh->x, sh->y, CL_DESTROYED);
                                            if(!check_field(sessions[id]->field1)){
                                                init_data_msg(&msg, &id, 1, MT_WIN, id);
                                                send_msg(sessions[id]->player2.sock, &msg);
                                                req.type = MT_LOSE;
                                            }  
                                            else{
                                                init_data_msg(&msg, &id, 1, MT_HIT, id);
                                                send_msg(sessions[id]->player2.sock, &msg);
                                                req.type = MT_HIT;
                                            }
                                        }
                                        else{
                                            printf("miss: %d-%d\n", sh->x, sh->y);
                                            set_cell(sessions[id]->field1, sh->x, sh->y, CL_MISS);

                                            init_data_msg(&msg, &id, 1, MT_MISS, id);
                                            send_msg(sessions[id]->player2.sock, &msg);
                                            req.type = MT_MISS;
                                        }
                                        close_msg(&msg);

                                        send_msg(sessions[id]->player1.sock, &req);
                                    }
                                    int ended = 0;
                                    if(!check_field(sessions[id]->field2)){
                                        printf("player1 won\n");
                                        end_session(sessions[id], 0);
                                        ended = 1;
                                        
                                    }
                                    if(!check_field(sessions[id]->field1)){
                                        printf("player2 won\n");
                                        end_session(sessions[id], 1);
                                        ended = 1;
                                    }

                                    if(ended){
                                        free(sessions[id]);
                                        sessions[id] = NULL;
                                    }
                                }
                                break;
                            case MT_RATING:{
                                    printf("requested rating\n");
                                    msg_t msg;
                                    init_data_msg(&msg, base.data, base.count * sizeof(record), MT_RATING, 0);
                                    send_msg(client_socks[i], &msg);
                                    close_msg(&msg);
                                }
                            default:
                                break;
                        }
                    }
                }
            }
        }
    }

    msg_t msg;

    for(int i = 0; i < MAX_SESSIONS_C; i++){
        if (sessions[i] != NULL){
            init_data_msg(&msg, base.data, base.count * sizeof(record), MT_ERROR, 0);
            send_msg(sessions[i]->player1.sock, &msg);
            send_msg(sessions[i]->player2.sock, &msg);
            close_msg(&msg);
        }
    }

    printf("releasing resources\n");
    vector_destroy(&queue);
    for(int i = 0; i < MAX_SESSIONS_C; i++)
        if(sessions[i] != NULL)
            free(sessions[i]);

    for(int i = 0; i < MAX_CLIENT_C; i++){
        if(client_socks[i] != 0){
            close(client_socks[i]);
            client_socks[i] = 0;
        }
    }
    close(master_sock);

    printf("saving data\n");
    save(argv[2], &base);
}