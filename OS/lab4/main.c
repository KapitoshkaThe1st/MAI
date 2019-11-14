#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>

#define SHM_FILE_NAME "/tmp_memory" // name of shared memory file (full path: /dev/shm/tmp_memory)
#define MAX_EXPR_L 128 // max length of expression
#define MAX_DIGS_C 11  // max digits count for int4
#define SIZE MAX_EXPR_L+2*sizeof(sem_t) // extra memory for 2 semaphores

int fetch_int(char *str, int *num);

int main(void)
{
	int fd = shm_open(SHM_FILE_NAME, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR); // создаем разделяемую память
	if(fd == -1){
		perror("shm::open_fail");
		exit(-1);
	}
	if(ftruncate(fd, SIZE) == -1){ // задаем ей нужный размер (для разделяемой памяти так можно, для обычных файлов -- не всегда. Для низ лучше lseek, write)
		perror("trucate::fail");
		exit(-1);
	}

	char *mapped_memory = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);	// отображаем на вирт. память
	if(mapped_memory == MAP_FAILED){
		perror("mmap::mapping_fail");
		fprintf(stderr, "%p", mapped_memory);
		exit(-1);
	}
	close(fd);	// дескриптор больше не нужен

	int shift = 0;
	sem_t *ready_to_print = (sem_t*)(mapped_memory + shift);	// семафор готовность к печати результата
	shift += sizeof(sem_t);
	sem_init(ready_to_print, 1, 0);

	sem_t *ready_to_comp = (sem_t*)(mapped_memory + shift);		// семафор готовность к рассчету
	shift += sizeof(sem_t);
	sem_init(ready_to_comp, 1, 0);

	char *buffer = (char *)(mapped_memory + shift);		// указатель на буффер, гду уже сами данные

	while (1)
	{
		pid_t child = fork();

		if (child == -1)
		{
			perror("fork");
			exit(-1);
		}
		else if (child > 0)
		{
			/* Parent process */
			// printf("Parent process: pid %d\n", getpid());

			char expr[MAX_EXPR_L];
			printf("%s\n", "Please enter an expression with / or * operations only");
			printf("%s\n", "Enter 'exit' to exit");
			scanf("%s", expr);
			printf("Input expression: %s\n", expr);
			if (!strcmp(expr, "exit"))
			{
				kill(child, SIGTERM);
				waitpid(child, NULL, 0);
				break;
			}
			
			int len = strlen(expr) + 1;
			int res;

			sprintf(buffer, "%s", expr);
			sem_post(ready_to_comp);
			sem_wait(ready_to_print);
			sscanf(buffer, "%d", &res);

			int status;
			waitpid(child, &status, 0);
			if (WIFSIGNALED(status))
			{
				perror("child::signalled");
				fprintf(stderr, "signal: %d\n", WTERMSIG(status));
				exit(-1);
			}
			else if (WIFEXITED(status))
			{
				char reason = WEXITSTATUS(status);
				if (reason != 0)
				{
					perror("child::exited");
					fprintf(stderr, "status: %d\n", reason);
					exit(-1);
				}
			}
			printf("Result: %d\n", res);
		}
		else
		{
			/* Child process */
			// printf("Child process: pid %d\n", getpid());
			// sleep(30);

			char expr2comp[MAX_EXPR_L];

			int res = 0;
			int i = 0;
			int operand;
			char sign;

			sem_wait(ready_to_comp);
			sscanf(buffer, "%s", expr2comp);

			i += fetch_int(expr2comp + i, &res);
			while (expr2comp[i] != '\0')
			{
				sign = expr2comp[i];
				++i;
				i += fetch_int(expr2comp + i, &operand);

				if (sign == '*')
				{
					res *= operand;
				}
				else
				{
					if (operand == 0)
					{
						perror("computation::division_by_zero");
						exit(-1);
					}
					res /= operand;
				}
			}
			sprintf(buffer, "%d", res);
			sem_post(ready_to_print);
			return 0;
		}
	}
	if (shm_unlink(SHM_FILE_NAME)){
		perror("shm::unlink::fail");
		exit(-1);
	}
	if(munmap(mapped_memory, 1000)){
		perror("mmap::munmap_failed");
		exit(-1);
	}
	sem_destroy(ready_to_comp);
	sem_destroy(ready_to_print);

	return 0;
}

/* fetch_int fetches int value from str to num and returns count of digs in num + 1 (it's neccessary for further fetches)*/
int fetch_int(char *str, int *num)
{
	int k = 0;
	char temp[MAX_DIGS_C] = {0};

	while (isdigit(str[k]) || str[k] == '-')
	{
		temp[k] = str[k];
		++k;
	}
	*num = atoi(temp);
	return k;
}
