#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

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
#define N 8000L
#define T 1000L
#endif

#define NUM_FP_OPS 10

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

  int t1, t2, t3, t4, t5, t6;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if ((N >= 1) && (T >= 1)) {
  for (t1=0;t1<=T-1;t1++) {
    lbp=0;
    ubp=floord(N-1,1024);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=floord(N-1,32);t4++) {
        for (t5=1024*t3;t5<=min(N-1,1024*t3+1023);t5++) {
          lbv=32*t4;
          ubv=min(N-1,32*t4+31);
#pragma ivdep
#pragma vector always
          for (t6=lbv;t6<=ubv;t6++) {
            A[1][t5][t6] = (exp(-0.001*t1)/4)*((t5-1) < 0 ? A[0][N-1][t6]: A[0][t5 - 1][t6]) + (exp(-0.001*t1)/4)*((t6+1) >= N ? A[0][t5][0] : A[0][t5][t6 + 1]) + (exp(-0.001*t1)/4)*((t5+1) >= N ? A[0][0][t6] : A[0][t5 + 1][t6]) + (exp(-0.001*t1)/4)*((t6-1) < 0 ? A[0][t5][N-1] : A[0][t5][t6 - 1]) + (1-exp(-0.001*t1))*A[0][t5][t6];;
          }
        }
      }
    }
    lbp=0;
    ubp=floord(N-1,1024);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=floord(N-1,32);t4++) {
        for (t5=1024*t3;t5<=min(N-1,1024*t3+1023);t5++) {
          lbv=32*t4;
          ubv=min(N-1,32*t4+31);
#pragma ivdep
#pragma vector always
          for (t6=lbv;t6<=ubv;t6++) {
            A[0][t5][t6] = (exp(-0.001*t1)/4)*((t5-1) < 0 ? A[1][N-1][t6]: A[1][t5 - 1][t6]) + (exp(-0.001*t1)/4)*((t6+1) >= N ? A[1][t5][0] : A[1][t5][t6 + 1]) + (exp(-0.001*t1)/4)*((t5+1) >= N ? A[1][0][t6] : A[1][t5 + 1][t6]) + (exp(-0.001*t1)/4)*((t6-1) < 0 ? A[1][t5][N-1] : A[1][t5][t6 - 1]) + (1-exp(-0.001*t1))*A[1][t5][t6];;
          }
        }
      }
    }
  }
}

#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken =  %7.5lfs\n", tdiff );
    printf("|MFLOPS =  %f\n", ((((double)NUM_FP_OPS * N *N *  T) / tdiff) / 1000000L));
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
