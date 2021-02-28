%%cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
__device__ int d_size;

__global__ void partition (int *arr, int *arr_l, int *arr_h,long int n)
{
    int z = blockIdx.x*blockDim.x+threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z<n)
      {
        int h = arr_h[z];
        int l = arr_l[z];
        int x = arr[h];
        int i = (l - 1);
        int temp;
        for (int j = l; j <= h- 1; j++)
          {
            if (arr[j] <= x)
              {
                i++;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
          }
        temp = arr[i+1];
        arr[i+1] = arr[h];
        arr[h] = temp;
        int p = (i + 1);
        if (p-1 > l)
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = l;
            arr_h[ind] = p-1;  
          }
        if ( p+1 < h )
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = p+1;
            arr_h[ind] = h; 
          }
      }
}
 
void quickSortIterative (int arr[],long int l,long int h)
{
    int lstack[ h - l + 1 ], hstack[ h - l + 1];
 
    int *d_d, *d_l, *d_h;
    long int top = -1;
 
    lstack[ ++top ] = l;
    hstack[ top ] = h;

    cudaMalloc(&d_d, (h-l+1)*sizeof(int));
    cudaMemcpy(d_d, arr,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_l, (h-l+1)*sizeof(int));
    cudaMemcpy(d_l, lstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_h, (h-l+1)*sizeof(int));
    cudaMemcpy(d_h, hstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);
    int n_t = 1;
    int n_b = 1;
    long int n_i = 1; 
    while ( n_i > 0 )
    {
        partition<<<n_b,n_t>>>( d_d, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost); 
        if (answer < 1024)
          {
            n_t = answer;
          }
        else
          {
            n_t = 1024;
            n_b = answer/n_t + (answer%n_t==0?0:1);
          }
        n_i = answer;
        cudaMemcpy(arr, d_d,(h-l+1)*sizeof(int),cudaMemcpyDeviceToHost);
    }
}
 

 
int main()
{
    long int n=1024*1;
    int arr[n];
    srand(time(NULL));
    for (int i = 0; i<n; i++)
       {
         arr[i] = rand ()%10000;
       }
    n = sizeof( arr ) / sizeof( *arr );
    cudaEvent_t start,end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
    quickSortIterative( arr, 0, n - 1 );
    cudaEventRecord(end);
		cudaEventSynchronize(end);
		float milliseconds=0;
		cudaEventElapsedTime(&milliseconds,start,end);
		double timeTaken=(double)milliseconds;
    double throughput = (n*sizeof(int))/(timeTaken);
    printf("%f,%f",timeTaken/1000,throughput);
    return 0;
}