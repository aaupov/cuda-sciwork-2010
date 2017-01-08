#include "header.h"

/*GPU kernels*/

/*Kernel_0 is non-optimized kernel*/
__global__ static void Kernel_0(int *a){
	extern __shared__ int a_d[]; //declaration of array in shared memory
	//Load element
	int threadId = threadIdx.x; 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	a_d[threadId] = a[i]; //each thread loads one element from global to shared memory									
	__syncthreads();
	//Compute
	for (int j=1; j<blockDim.x; j*=2){
		if ((threadId % (2*j)) == 0) a_d[threadId]+=a_d[threadId+j];
		__syncthreads();
	}		
	//Write result
	if (threadId == 0) a[blockIdx.x] = a_d[0]; //write to global memory from shared
}

/*Kernel_1 makes reduction non-divergent*/
__global__ static void Kernel_1(int *a){
	extern __shared__ int a_d[];
	int threadId = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadId;
	a_d[threadId] = a[i];
	__syncthreads();
	for (int j=1; j<blockDim.x; j*=2){ //just make reduction non-divergent 
		int idx = 2*j*threadId; 
		if (idx<blockDim.x) a_d[idx]+=a_d[idx+j];
		__syncthreads();
	}
	if (!threadId) a[blockIdx.x] = a_d[0];
}

/*Kernel_2 eliminates memory bank conflicts due to sequental addressing*/
__global__ static void Kernel_2(int *a){
	extern __shared__ int a_d[];
	int threadId = threadIdx.x;
	UINT i = blockIdx.x*blockDim.x + threadId;
	a_d[threadId] = a[i];
	__syncthreads();
	for (int j = blockDim.x>>1; j>0; j>>=1){ //another type of indexing
		if (threadId < j) a_d[threadId]+=a_d[threadId+j];
		__syncthreads();
	}
	if (!threadId) a[blockIdx.x] = a_d[0];
}

/*Kernel_3 eliminates idle threads and makes first reduction while loading from global memory*/
__global__ static void Kernel_3(int *a){
	extern __shared__ int a_d[];
	int threadId = threadIdx.x;
	UINT i = blockIdx.x*blockDim.x*2+threadId;//halve number of blocks
	a_d[threadId]=a[i];
	a_d[threadId]+=a[i+blockDim.x];//and make first step of reduction while load
	__syncthreads();
	for (int j = blockDim.x>>1; j>0; j>>=1){ 
		if (threadId < j) a_d[threadId]+=a_d[threadId+j];
		__syncthreads();
	}
	if (!threadId) a[blockIdx.x] = a_d[0];
}

/*Kernel_4 unrolls last series of threads in block*/
__global__ static void Kernel_4(int *a){
	extern __shared__ int a_d[];
	int threadId = threadIdx.x;
	UINT i = blockIdx.x*blockDim.x*2+threadId;
	a_d[threadId]=a[i];
	a_d[threadId]+=a[i+blockDim.x];
	__syncthreads();
	for (int j = blockDim.x>>1; j>32; j>>=1){ 
		if (threadId < j) a_d[threadId]+=a_d[threadId+j];
		__syncthreads();
	}
	if (threadId<32){ //unroll to avoid useless __syncthreads()
		a_d[threadId]+=a_d[threadId+32];
		a_d[threadId]+=a_d[threadId+16];
		a_d[threadId]+=a_d[threadId+8];
		a_d[threadId]+=a_d[threadId+4];
		a_d[threadId]+=a_d[threadId+2];
		a_d[threadId]+=a_d[threadId+1];
	}
	if (!threadId) a[blockIdx.x] = a_d[0];
}

/*Kernel_5 completely unrolled*/
__global__ static void Kernel_5(int *a, int blockSize){
	extern __shared__ int a_d[];
	int threadId = threadIdx.x;
	UINT i = blockIdx.x*blockDim.x*2+threadId;
	a_d[threadId]=a[i];
	a_d[threadId]+=a[i+blockSize];
	__syncthreads();
	if (blockSize==512){ 
		if (threadId<256) a_d[threadId]+=a_d[threadId+256]; 
		__syncthreads();
	}
	if (blockSize>=256){ 
		if (threadId<128) a_d[threadId]+=a_d[threadId+128]; 
		__syncthreads();
	}
	if (blockSize>=128){ 
		if (threadId<64) a_d[threadId]+=a_d[threadId+64]; 
		__syncthreads();
	}
	if (threadId<32){ //if less than warp size __syncthreads() isn't needed
		if (blockSize>=64) a_d[threadId]+=a_d[threadId+32];
		if (blockSize>=32) a_d[threadId]+=a_d[threadId+16];
		if (blockSize>=16) a_d[threadId]+=a_d[threadId+8];
		if (blockSize>=8) a_d[threadId]+=a_d[threadId+4];
		if (blockSize>=4) a_d[threadId]+=a_d[threadId+2];
		if (blockSize>=2) a_d[threadId]+=a_d[threadId+1];
	}
	if (!threadId) a[blockIdx.x] = a_d[0];
}

/*Kernel_6 Brent theorem optimization - multiple adds per thread*/
__global__ static void Kernel_6(int *a, UINT n, int blockSize){
	extern __shared__ int a_d[];
	UINT threadId = threadIdx.x;
	UINT i = blockIdx.x*blockSize*2+threadId;
	UINT grid = blockSize*gridDim.x*2; //get num of active blocks
	a_d[threadId]=0;
	while(i<n){
		a_d[threadId]+=a[i]+a[i+blockSize]; //two adds
		i+=grid; //
	};
	__syncthreads();
	if (blockSize>=512){ 
		if (threadId<256) a_d[threadId]+=a_d[threadId+256]; 
		__syncthreads();
	}
	if (blockSize>=256){ 
		if (threadId<128) a_d[threadId]+=a_d[threadId+128]; 
		__syncthreads();
	}
	if (blockSize>=128){ 
		if (threadId<64) a_d[threadId]+=a_d[threadId+64]; 
		__syncthreads();
	}
	if (threadId<32){ //if less than warp size __syncthreads() isn't needed
		if (blockSize>=64) a_d[threadId]+=a_d[threadId+32];
		if (blockSize>=32) a_d[threadId]+=a_d[threadId+16];
		if (blockSize>=16) a_d[threadId]+=a_d[threadId+8];
		if (blockSize>=8) a_d[threadId]+=a_d[threadId+4];
		if (blockSize>=4) a_d[threadId]+=a_d[threadId+2];
		if (blockSize>=2) a_d[threadId]+=a_d[threadId+1];
	}
	if (!threadId) a[blockIdx.x] = a_d[0];
}

/*CPU*/

//Command-line syntax described in readme.txt
int main(int argc, char* argv[]){ 
	int shift, debug,info,pause,maxThreads;
	if (argc>1){ 
		shift=atoi(argv[1]);
		debug=atoi(argv[2]);
		info=atoi(argv[3]);
		pause=atoi(argv[4]);
		maxThreads=atoi(argv[5]);
	}
	else { 
		shift=22;
		debug=0;
		info=0;
		pause=1;
		maxThreads=0;
	}

	/*Device initialization*/
	int count = 0, i = 0;	
	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	cudaDeviceProp prop;
	for(i = 0; i < count; i++) { //select first CUDA v1.x device 
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) break;
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no CUDA devices\n");
		return false;
	}
	cudaSetDevice(i);
	printf("%s\n",prop.name);
	if (info){
		int procs = prop.multiProcessorCount * 8;
		float clock = (float) prop.clockRate / 1000000;
		printf("%d processors @ %.2fGHz\n",procs,clock);
	}
	/*Preparing to call GPU kernel*/
	int *a_h, *a_d, *a_g;			//arrays for host and device
	UINT n = 1<<shift;				//read main() description
	size_t size = sizeof(int)*n;
	if (info) printf("Size==%d==2^%d\n",n,shift);
	a_h = (int*)malloc(size);		//allocate array at host
	cudaMalloc((void**) &a_d, size);//allocate array at device
	a_g = (int*)malloc(size);		//allocate array for gpu result
	srand(time(NULL));				//randomize
	for (i=0;i<n;i++){				//fill array at host with small ints
		a_h[i]=(rand() & 0xFE) - 0x7F; //-127..127
		if (debug){
			printf("%5d",a_h[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device
	long int cpu = 0, gpu = 0;
	UINT cpu_timer = 0;
	cutCreateTimer(&cpu_timer);
	cutStartTimer(cpu_timer);
	for (int j=0;j<n;j++){ //calculate sum in source array
		cpu+=a_h[j];
	}
	cutStopTimer(cpu_timer);
	if (info){
		float cpu_time = cutGetTimerValue(cpu_timer);
		printf("cpu time: %f ms\n",cpu_time);
	}

	/*Computing blocks\threads grid*/
	if (!maxThreads) maxThreads = prop.maxThreadsPerBlock; //gather information from device
	int maxBlocks = prop.maxGridSize[0];
	int threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
	int blocks = n/threads;
	dim3 dimBlock(threads, 1);
	dim3 dimGrid(blocks, 1); 
    int memSize = threads * sizeof(int);
	if (info) printf("Blocks:%d Threads:%d\n",blocks,threads);

	UINT timer = 0; 
	///*Compute on GPU Kernel_0*/	
	gpu=0;
	float timer0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_0 <<<dimGrid, dimBlock, memSize>>> (a_d); //launch kernel
	CUT_CHECK_ERROR("Kernel_0 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost); //copy array to host
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_0\n"); 
	else printf("\nSomething wrong with Kernel_0\n");
	if (info){ 
		timer0 = cutGetTimerValue(timer);//get timer value
		float bw0 = size/timer0/1000000;//calculate bandwidth
		printf("Time: %f (ms)\n%.3fGB/s\n", 
			timer0, bw0); 
	}
	cutDeleteTimer(timer); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device

	///*Compute on GPU Kernel_1*/ 
	gpu=0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_1<<<dimGrid, dimBlock, memSize>>>(a_d); //launch kernel
	CUT_CHECK_ERROR("Kernel_1 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost); //copy array to host
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_1\n"); 
	else printf("\nSomething wrong with Kernel_1\n");
	if (info){ 
		float timer1 = cutGetTimerValue(timer);//get timer value
		float bw1 = size/timer1/1000000;
		float sp_up = timer0/timer1;//speedup 
		printf("Processing time: %f (ms)\n%.3fGB/s\n", timer1, bw1);
		printf("Speedup: %.2fx\n",sp_up);
	}
	cutDeleteTimer(timer); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device

	///*Compute on GPU Kernel_2*/ 
	gpu=0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_2<<<dimGrid, dimBlock, memSize>>>(a_d); //launch kernel
	CUT_CHECK_ERROR("Kernel_2 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost);
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_2\n"); 
	else printf("\nSomething wrong with Kernel_2\n");
	if (info){ 
		float timer2 = cutGetTimerValue(timer);//get timer value
		float bw2 = size/timer2/1000000;
		float sp_up = timer0/timer2;//speedup 
		printf("Processing time: %f (ms)\n%.3fGB/s\n", timer2, bw2);
		printf("Speedup: %.2fx\n",sp_up);
	}
	CUT_SAFE_CALL(cutDeleteTimer(timer)); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device

	///*Compute on GPU Kernel_3*/ 
	//this kernel requires different num of threads\blocks, look in Kernel_3
	if (n<maxThreads*2) threads = n / 2;
	else threads = maxThreads;
	if (n==1) threads=1;
    blocks = n / (threads*2);
	dim3 dimBlock_3(threads, 1);
	dim3 dimGrid_3(blocks, 1); 
    int memSize_3 = threads * sizeof(int);
	gpu=0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_3<<<dimGrid_3, dimBlock_3, memSize_3>>>(a_d); //launch kernel
	CUT_CHECK_ERROR("Kernel_3 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost);
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_3\n"); 
	else printf("\nSomething wrong with Kernel_3\n");
	if (info){ 
		float timer3 = cutGetTimerValue(timer);//get timer value
		float bw3 = size/timer3/1000000;
		float sp_up = timer0/timer3;//speedup 
		printf("Processing time: %f (ms)\n%.3fGB/s\n", timer3, bw3);
		printf("Speedup: %.2fx\n",sp_up);
	}
	cutDeleteTimer(timer); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device
	
	///*Compute on GPU Kernel_4*/ 
	//the same run parameters as Kernel_3
	gpu=0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_4<<<dimGrid_3, dimBlock_3, memSize_3>>>(a_d); //launch kernel
	CUT_CHECK_ERROR("Kernel_4 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost);
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_4\n"); 
	else printf("\nSomething wrong with Kernel_4\n");
	if (info){ 
		float timer4 = cutGetTimerValue(timer);//get timer value
		float bw4 = size/timer4/1000000;
		float sp_up = timer0/timer4;//speedup 
		printf("Processing time: %f (ms)\n%.3fGB/s\n", timer4, bw4);
		printf("Speedup: %.2fx\n",sp_up);
	}
	cutDeleteTimer(timer); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device
	
	///*Compute on GPU Kernel_5*/ 
	//the same run parameters as Kernel_3
	gpu=0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_5<<<dimGrid_3, dimBlock_3, memSize_3 >>>(a_d, threads);
	CUT_CHECK_ERROR("Kernel_5 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost);
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_5\n"); 
	else printf("\nSomething wrong with Kernel_5\n");
	if (info){ 
		float timer5 = cutGetTimerValue(timer);//get timer value
		float bw5 = size/timer5/1000000;
		float sp_up = timer0/timer5;//speedup 
		printf("Processing time: %f (ms)\n%.3fGB/s\n", timer5, bw5);
		printf("Speedup: %.2fx\n",sp_up);
	}
	cutDeleteTimer(timer); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device

	/*Compute on GPU Kernel_6*/ 
	int o = shift;
	threads = (n < maxThreads*o) ? n/o : maxThreads;
    blocks = n / (threads * o);
	blocks = min(maxBlocks,blocks);
	if (info) printf("\nKernel_6 Blocks: %d Threads: %d\n",blocks,threads);
	dim3 dimBlock_6(threads, 1);
	dim3 dimGrid_6(blocks, 1); 
    int memSize_6 = threads * sizeof(int);
	gpu=0;
	cutCreateTimer(&timer); //create timer for computation kernel execution time
	cutStartTimer(timer); //start timer
	Kernel_6<<<dimGrid_6, dimBlock_6, memSize_6 >>>(a_d,n,threads);
	CUT_CHECK_ERROR("Kernel_6 execution failed"); //if there are something wrong
	cudaThreadSynchronize();
	cutStopTimer(timer); //stop timer

	cudaMemcpy(a_g, a_d, size, cudaMemcpyDeviceToHost);
	for (i=0; i<blocks; i++){ //if blocks>1 we need to collect partial results from each block
		gpu+=a_g[i];
		if (debug){
			printf("%6d",a_g[i]);
			if ((i+1)%16==0) printf("\n");
		}
	}
	if (cpu==gpu) printf("\nSuccessfully completed on Kernel_6\n"); 
	else printf("\nSomething wrong with Kernel_6\n");
	if (info){ 
		float timer6 = cutGetTimerValue(timer);//get timer value
		float bw6 = size/timer6/1000000;
		float sp_up = timer0/timer6;//speedup 
		printf("Processing time: %f (ms)\n%.3fGB/s\n", timer6, bw6);
		printf("Speedup: %.2fx\n",sp_up);
	}
	cutDeleteTimer(timer); //delete timer
	if (debug) printf("cpu:%d gpu:%d\n\n",cpu,gpu);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);//copy array to device

	printf("\n\n");
	//Free mem
	cudaFree(a_d);
	free(a_h);
	free(a_g);

	//if (pause) system("pause");
	return 0;
}
