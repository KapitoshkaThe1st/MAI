
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// перевод из трехмерной индексации к одномерной, используется в вызовах send\recv
#define block_index(i, j, k) ((k) * (n_blocks_x * n_blocks_y)) + (j) * n_blocks_x + (i)

// перевод из одномерной индексации в трехмерную
#define block_index_i(n) ((n) % (n_blocks_x * n_blocks_y) % n_blocks_x)
#define block_index_j(n) ((n) % (n_blocks_x * n_blocks_y) / n_blocks_x)
#define block_index_k(n) ((n) / (n_blocks_x * n_blocks_y))

// перевод из трехмерной индексации к одномерной, используется в вызовах send\recv
#define cell_index(i, j, k) (((k) + 1) * ((block_size_x + 2) * (block_size_y + 2)) + ((j) + 1) * (block_size_x + 2) + ((i) + 1)) 

// перевод из одномерной индексации в трехмерную
#define cell_index_i(n) ((n) % ((block_size_x + 2) * (block_size_y + 2)) % (block_size_x + 2) - 1)
#define cell_index_j(n) ((n) % ((block_size_x + 2) * (block_size_y + 2)) / (block_size_x + 2) - 1)
#define cell_index_k(n) ((n) / ((block_size_x + 2) * (block_size_y + 2)) - 1)

#define LEFT 0
#define RIGHT 1
#define BACK 2
#define FRONT 3
#define DOWN 4
#define UP 5

double compute(double *u_new, double *u, int block_size_x, int block_size_y, int block_size_z, double hx, double hy, double hz){
	double max_error = 0.0;
	for(int i = 0; i < block_size_x; ++i)
		for(int j = 0; j < block_size_y; ++j)
			for(int k = 0; k < block_size_z; ++k){
				double inv_hxsqr = 1.0 / (hx * hx);
				double inv_hysqr = 1.0 / (hy * hy);
				double inv_hzsqr = 1.0 / (hz * hz);

				double num = (u[cell_index(i+1, j, k)] + u[cell_index(i-1, j, k)]) * inv_hxsqr
					+ (u[cell_index(i, j+1, k)] + u[cell_index(i, j-1, k)]) * inv_hysqr
					+ (u[cell_index(i, j, k+1)] + u[cell_index(i, j, k-1)]) * inv_hzsqr;
				double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);
				double temp = num / denum;
				double error = fabs(u[cell_index(i, j, k)] - temp);

				if(error > max_error)
					max_error = error;

				u_new[cell_index(i, j, k)] = temp;
			}

	return max_error;
}

int main(int argc,char **argv)
{
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int n_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

	int n_blocks_x, n_blocks_y, n_blocks_z;
	int block_size_x, block_size_y, block_size_z;
	double eps;
	double lx, ly, lz;
	double u_down, u_up, u_left, u_right, u_front, u_back, u0;

	char *output_file_path = NULL;
	if(rank == 0){
		assert(scanf("%d %d %d", &n_blocks_x, &n_blocks_y, &n_blocks_z) == 3);
		assert(scanf("%d %d %d", &block_size_x, &block_size_y, &block_size_z) == 3);

		output_file_path = (char*)malloc(PATH_MAX * sizeof(char));
		assert(scanf("%s", output_file_path) == 1);
		assert(scanf("%lf", &eps) == 1);
		assert(scanf("%lf %lf %lf", &lx, &ly, &lz) == 3);
		assert(scanf("%lf %lf %lf %lf %lf %lf %lf", &u_down, &u_up, &u_left, &u_right, &u_front, &u_back, &u0) == 7);

		fprintf(stderr, "n_blocks_x: %d n_blocks_y: %d n_blocks_z: %d ", n_blocks_x, n_blocks_y, n_blocks_z);

		fprintf(stderr, "block_size_x: %d block_size_y: %d block_size_z: %d ", block_size_x, block_size_y, block_size_z);

		fprintf(stderr, "eps: %e ", eps);

		fprintf(stderr, "lx: %lf ly: %lf lz: %lf ", lx, ly, lz);

		fprintf(stderr, "u_down: %lf ", u_down);
		fprintf(stderr, "u_up: %lf ", u_up);
		fprintf(stderr, "u_left: %lf ", u_left);
		fprintf(stderr, "u_right: %lf ", u_right);
		fprintf(stderr, "u_front: %lf ", u_front);
		fprintf(stderr, "u_back: %lf ", u_back);
		fprintf(stderr, "u0: %lf\n", u0);

	}
	
	MPI_Status status;

	MPI_Bcast(&n_blocks_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_blocks_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_blocks_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_size_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_size_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_size_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double hx = lx / (n_blocks_x * block_size_x), hy = ly / (n_blocks_y * block_size_y), hz = lz / (n_blocks_z * block_size_z);

	int block_i = block_index_i(rank), block_j = block_index_j(rank), block_k = block_index_k(rank);

	MPI_Barrier(MPI_COMM_WORLD);

	int n_cells_per_block = (block_size_x + 2) * (block_size_y + 2) * (block_size_z + 2);
	double *u = (double*)malloc(sizeof(double) * n_cells_per_block);
	double *u1 = (double*)malloc(sizeof(double) * n_cells_per_block);

	for(int i = 0; i < n_cells_per_block; ++i)
		u[i] = u0;

	int max_dim = fmax(block_size_x, fmax(block_size_y, block_size_z));
	double *buffer = (double*)malloc(sizeof(double) * max_dim * max_dim);
	double *buffer1 = (double*)malloc(sizeof(double) * max_dim * max_dim);

	double max_error = 100.0;

	while(max_error > eps){
		if(block_i > 0){
			// отсылка и прием левого граничного условия
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					buffer[j * block_size_z + k] = u[cell_index(0, j, k)];

			int count = block_size_y * block_size_z;
			int exchange_process_rank = block_index(block_i-1, block_j, block_k);

			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, LEFT,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
				MPI_COMM_WORLD, &status);
				
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(-1, j, k)] = buffer1[j * block_size_z + k];
		}
		else{
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(-1, j, k)] = u_left;
		}

		if(block_i < n_blocks_x-1){
			// отсылка и прием правого граничного условия
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					buffer[j * block_size_z + k] = u[cell_index(block_size_x-1, j, k)];

			int count = block_size_y * block_size_z;
			int exchange_process_rank = block_index(block_i+1, block_j, block_k);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, LEFT,
				MPI_COMM_WORLD, &status);

			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(block_size_x, j, k)] = buffer1[j * block_size_z + k];
		}
		else{
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(block_size_x, j, k)] = u_right;
		}

		if(block_j > 0){
			// отсылка и прием переднего граничного условия
			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					buffer[i * block_size_z + k] = u[cell_index(i, 0, k)];

			int count = block_size_x * block_size_z;
			int exchange_process_rank = block_index(block_i, block_j-1, block_k);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, FRONT,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, BACK,
				MPI_COMM_WORLD, &status);

			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(i, -1, k)] = buffer1[i * block_size_z + k];
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(i, -1, k)] = u_front;
		}

		if(block_j < n_blocks_y-1){
			// отсылка и прием заднего граничного условия
			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					buffer[i * block_size_z + k] = u[cell_index(i, block_size_y-1, k)];

			int count = block_size_x * block_size_z;
			int exchange_process_rank = block_index(block_i, block_j+1, block_k);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, BACK,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, FRONT,
				MPI_COMM_WORLD, &status);

			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(i, block_size_y, k)] = buffer1[i * block_size_z + k];
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					u[cell_index(i, block_size_y, k)] = u_back;
		}

		if(block_k > 0){
			// отсылка и прием нижнего граничного условия
			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					buffer[i * block_size_y + j] = u[cell_index(i, j, 0)];

			int count = block_size_x * block_size_y;
			int exchange_process_rank = block_index(block_i, block_j, block_k-1);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, DOWN,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, UP,
				MPI_COMM_WORLD, &status);

			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					u[cell_index(i, j, -1)] = buffer1[i * block_size_y + j];
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					u[cell_index(i, j, -1)] = u_down;
		}

		if(block_k < n_blocks_z-1){
			// отсылка и прием верхнего граничного условия
			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					buffer[i * block_size_y + j] = u[cell_index(i, j, block_size_z-1)];

			int count = block_size_x * block_size_y;
			int exchange_process_rank = block_index(block_i, block_j, block_k+1);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, UP,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, DOWN,
				MPI_COMM_WORLD, &status);

			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					u[cell_index(i, j, block_size_z)] = buffer1[i * block_size_y + j];
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					u[cell_index(i, j, block_size_z)] = u_up;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		double error = compute(u1, u, block_size_x, block_size_y, block_size_z, hx, hy, hz);

		double *temp = u1;
		u1 = u;
		u = temp;

		// вычисление максимальной ошибки
		MPI_Allreduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	int count = block_size_x;
	if(rank == 0){
		FILE *output_file = fopen(output_file_path, "w"); 

		for(int bk = 0; bk < n_blocks_z; ++bk){
			for(int k = 0; k < block_size_z; ++k){
				for(int bj = 0; bj < n_blocks_y; ++bj){
					for(int j = 0; j < block_size_y; ++j){
						for(int bi = 0; bi < n_blocks_x; ++bi){
							int block_idx = block_index(bi, bj, bk);
							if(block_idx == 0){
								for(int i = 0; i < block_size_x; ++i)
									buffer[i] = u[cell_index(i, j, k)];
							}
							else{
								MPI_Recv(buffer, count, MPI_DOUBLE, block_idx, k * block_size_z + j, MPI_COMM_WORLD, &status);
							}

							for(int i = 0; i < block_size_x; ++i){
								// fprintf(output_file, "(%d, %d, %d) ", bi * block_size_x + i + 1, bj * block_size_y + j + 1, bk * block_size_z + k + 1);
								fprintf(output_file, "%e ", buffer[i]);
							}
						}
						fprintf(output_file, "\n");
					}
				}
				fprintf(output_file, "\n");
			}
		}

		fclose(output_file);
	}
	else{
		for(int k = 0; k < block_size_z; ++k){
			for(int j = 0; j < block_size_y; ++j){
				for(int i = 0; i < block_size_x; ++i)
					buffer[i] = u[cell_index(i, j, k)];
				MPI_Send(buffer, count, MPI_DOUBLE, 0, k * block_size_z + j, MPI_COMM_WORLD);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	free(output_file_path);
	free(buffer);
	free(buffer1);
	free(u);
	free(u1);

    MPI_Finalize();	
}
