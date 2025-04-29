/*
 * Discretized 2D heat equation stencil with non periodic boundary conditions
 * Adapted from Pochoir test bench
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

/*
 * N is the number of points
 * T is the number of timesteps
 */
#ifdef HAS_DECLS
#include "decls.h"
#else
#define N 32768L
#define T 32768L
#endif

#define N 32768L

/* Define our arrays */
double A[2][N][N];
double total=0; double sum_err_sqr=0;
int chtotal=0;
int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;

        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }

    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;

        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    return x->tv_sec < y->tv_sec;
}

int main(int argc, char * argv[]) {
    long int t, i, j, k;
    const int BASE = 1024;

    // for timekeeping
    int ts_return = -1;
    struct timeval start, end, result;
    double tdiff = 0.0;

    printf("Number of points = %ld\t|Number of timesteps = %ld\t", N*N, T);

    /* Initialization */
    srand(42); // seed with a constant value to verify results

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[0][i][j] = 1.0 * (rand() % BASE);
        }
    }

#ifdef TIME
    gettimeofday(&start, 0);
#endif

#pragma scop
    for (t = 0; t < T; t++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[1][i][j] = (exp(-0.001*t)/4)*((i-1) < 0 ? 0: A[0][i - 1][j]) + (exp(-0.001*t)/4)*((j+1) >= N ? 0 : A[0][i][j + 1]) 
                                    + (exp(-0.001*t)/4)*((i+1) >= N ? 0 : A[0][i + 1][j]) + (exp(-0.001*t)/4)*((j-1) < 0 ? 0 : A[0][i][j - 1]) 
                                    + (1-exp(-0.001*t))*A[0][i][j];
            }
        }
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[0][i][j] = (exp(-0.001*t)/4)*((i-1) < 0 ? 0: A[1][i - 1][j]) + (exp(-0.001*t)/4)*((j+1) >= N ? 0 : A[1][i][j + 1]) 
                                    + (exp(-0.001*t)/4)*((i+1) >= N ? 0 : A[1][i + 1][j]) + (exp(-0.001*t)/4)*((j-1) < 0 ? 0 : A[1][i][j - 1]) 
                                    + (1-exp(-0.001*t))*A[1][i][j];
            }
        }
    }
#pragma endscop

#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken =  %7.5lfs\n", tdiff );
#endif

#ifdef VERIFY
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            total+= A[T%2][i][j] ;
        }
    }
    printf("|sum: %e\t", total);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum_err_sqr += (A[T%2][i][j] - (total/N))*(A[T%2][i][j] - (total/N));
        }
    }
    printf("|rms(A) = %7.2f\t", sqrt(sum_err_sqr));
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            chtotal += ((char *)A[T%2][i])[j];
        }
    }
    printf("|sum(rep(A)) = %d\n", chtotal);
#endif
    return 0;
}
