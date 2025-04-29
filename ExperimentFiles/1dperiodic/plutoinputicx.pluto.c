#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*
 * Discretized 1D heat equation stencil with non periodic boundary conditions
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
#define N 1600000L
#define T 3000000L
#endif

#define N 1600000L

/* Define our arrays */
double A[2][N];
double total=0; double sum_err_sqr=0;
long int chtotal=0;
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
    printf("The value of T is: %ld\n", T);
    // for timekeeping
    int ts_return = -1;
    struct timeval start, end, result;
    double tdiff = 0.0;
long count=0;
    printf("Number of points = %ld\t|Number of timesteps = %ld\t", N, T);

    /* Initialization */
    srand(42); // seed with a constant value to verify results

    for (i = 0; i < N; i++) {
        A[0][i] = 1.0 * (rand() % BASE);
    }

#ifdef TIME
    gettimeofday(&start, 0);
#endif
  int t1, t2, t3, t4;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if ((N >= 1) && (T >= 1)) {
  for (t1=0;t1<=T-1;t1++) {
    for (t3=0;t3<=N-1;t3++) {
      A[1][t3] = (pow(exp(1),-0.01*t1)/2)*((t3 == N-1)?A[0][0]:A[0][t3+1]) + (1-pow(exp(1),-0.01*t1))*A[0][t3] + (pow(exp(1),-0.01*t1)/2)*((t3 == 0)?A[1][N-1]:A[0][t3-1]);;
    }
    lbp=0;
    ubp=floord(N-1,2048);
#pragma omp parallel for private(lbv,ubv,t4)
    for (t3=lbp;t3<=ubp;t3++) {
      lbv=2048*t3;
      ubv=min(N-1,2048*t3+2047);
#pragma ivdep
#pragma vector always
      for (t4=lbv;t4<=ubv;t4++) {
        A[0][t4] = (pow(exp(1),-0.01*t1)/2)*((t4 == N-1)?A[1][0]:A[1][t4+1]) + (1-pow(exp(1),-0.01*t1))*A[1][t4] + (pow(exp(1),-0.01*t1)/2)*((t4 == 0)?A[1][N-1]:A[1][t4-1]);;
      }
    }
  }
}
printf("The value of T is: %ld\n", T);

#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken =  %7.5lfs\t", tdiff);
#endif

#ifdef VERIFY
    total=0;
    for (i = 0; i < N; i++) {
        total+= A[T%2][i] ;
    }
    printf("|sum: %e\t", total);
    for (i = 0; i < N; i++) {
        sum_err_sqr += (A[T%2][i] - (total/N))*(A[T%2][i] - (total/N));
    }
    printf("|rms(A) = %7.2f\t", sqrt(sum_err_sqr));
    for (i = 0; i < N; i++) {
        chtotal += ((char *)A[T%2])[i];
    }
    printf("|sum(rep(A)) = %ld\n", chtotal);
#endif
    return 0;
}

// icc -O3 -fp-model precise heat_1d_np.c -o op-heat-1d-np -lm
// /* @ begin PrimeTile (num_tiling_levels=1; first_depth=1; last_depth=-1; boundary_tiling_level=-1;) @*/
// /* @ begin PrimeRegTile (scalar_replacement=0; T1t3=8; T1t4=8; ) @*/
// /* @ end @*/
// ,t2,t3,t4,t5,t6)
