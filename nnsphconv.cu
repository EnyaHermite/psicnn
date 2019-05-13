#include "nnsphconv.hpp"
#include <cassert>
#include <stdlib.h>
#include <cuda.h>
#include <device_functions.h>

#define VL_CUDA_NUM_THREADS 1024

//#if __CUDA_ARCH__ >= 200
//#define VL_CUDA_NUM_THREADS 1024
//#else
//#define VL_CUDA_NUM_THREADS 512
//#endif

#define maxStreams 16 // maximum number of streams used for concurrency

using namespace std;

inline size_t divideAndRoundUp(size_t a, size_t b)
{
	return (a + b - 1) / b;
}

__global__ void
setToOnes(float *data, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x; // 1D grid of 1D blocks
	if (index < size) data[index] = 1;
}

__global__ void
setToZeros(float *data, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x; // 1D grid of 1D blocks
	if (index < size) data[index] = 0;
}

void createStreams(cudaStream_t *streams, int N)
{
	for (int i = 0; i < N; i++)
	{
		cudaStreamCreate(&(streams[i]));
	}
}

void destroyStreams(cudaStream_t *streams, int N)
{
	for (int i = 0; i < N; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
}

bool gemmBatched(cublasHandle_t &handle,
	float const *A, int ref_A, char chA, // left input
	float const *B, int ref_B, char chB, // right input
	float *C, int ref_C, // output C[] = (A[]) * (B[])
	int *filtPerNum, // filtPerNum is only required in the computation of derFilter
	unsigned int const *map,
	unsigned int Fin,
	unsigned int Fout,
	unsigned int Nin,
	unsigned int Nout,
	unsigned int Nfilt)
{
	bool status = true;
	int N = (Nin > Nout) ? Nin : Nout; // batch size
									   
	// prepare data arrays for matrix-matrix multiplication
	float **ptrA, **ptrB, **ptrC;
	ptrA = (float **)malloc(N * sizeof(*ptrA));
	ptrB = (float **)malloc(N * sizeof(*ptrB));
	ptrC = (float **)malloc(N * sizeof(*ptrC));

	unsigned int Size[] = { Fin, (Fin * Fout), Fout };
	if (!filtPerNum)
	{
		for (int node = 0; node < N; node++)
		{
			// assume input and map are correspondingly sorted
			ptrdiff_t A_Offset = Size[ref_A] * map[node * 4 + ref_A];
			ptrdiff_t B_Offset = Size[ref_B] * map[node * 4 + ref_B];
			ptrdiff_t C_Offset = Size[ref_C] * node;

			ptrA[node] = (float *)A + A_Offset;
			ptrB[node] = (float *)B + B_Offset;
			ptrC[node] = (float *)C + C_Offset;
		}
	}
	else
	{
		for (int idx = 0; idx < Nfilt; idx++)
		{
			filtPerNum[idx] = 0;
		} // initialize to zeros

		int iter = 0;
		for (int idx = 0; idx < Nfilt; idx++)
		{
			if (iter >= N) { break; }

			for (int node = 0; node < N; node++)
			{
				ptrdiff_t filterID = map[node * 4 + 1];				

				if (idx == filterID)
				{
					int a = map[node * 4 + ref_A];
					int b = map[node * 4 + ref_B];
					ptrdiff_t A_Offset = Size[ref_A] * map[node * 4 + ref_A];
					ptrdiff_t B_Offset = Size[ref_B] * map[node * 4 + ref_B];
					ptrdiff_t C_Offset = Size[ref_C] * iter;

					ptrA[iter] = (float *)A + A_Offset;
					ptrB[iter] = (float *)B + B_Offset;
					ptrC[iter] = (float *)C + C_Offset;

					filtPerNum[idx]++;
					iter++;					
				}
			}
		}
	}

	float **ptrA_dev, **ptrB_dev, **ptrC_dev;
	cudaError_t err1 = cudaMalloc((void **)&ptrA_dev, N * sizeof(*ptrA));
	cudaError_t err2 = cudaMalloc((void **)&ptrB_dev, N * sizeof(*ptrB));
	cudaError_t err3 = cudaMalloc((void **)&ptrC_dev, N * sizeof(*ptrC));
	if(err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess)
	{
		status = false;
		goto CLEANUP;
	}

	err1 = cudaMemcpy(ptrA_dev, ptrA, N * sizeof(*ptrA), cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(ptrB_dev, ptrB, N * sizeof(*ptrB), cudaMemcpyHostToDevice);
	err3 = cudaMemcpy(ptrC_dev, ptrC, N * sizeof(*ptrC), cudaMemcpyHostToDevice);
	if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess)
	{
		status = false;
		goto CLEANUP;
	}

	float alpha, beta;
    alpha = 1, beta = 0; // constant settings
	cublasOperation_t opt_A, opt_B;

	int rows[3];
    rows[0] = Fin;
    rows[1] = Fin;
    rows[2] = Fout;
	int cols[3];
    cols[0] = 1;
    cols[1] = Fout;
    cols[2] = 1;

	int m, n, k;
	if (chA == 'n' || chA == 'N')
	{
		m = rows[ref_A];
		k = cols[ref_A];
		opt_A = CUBLAS_OP_N;
	}
	else
	{
		m = cols[ref_A];
		k = rows[ref_A];
		opt_A = CUBLAS_OP_T;
	}
	if (chB == 'n' || chB == 'N')
	{
		n = cols[ref_B];
		opt_B = CUBLAS_OP_N;
	}
	else
	{
		n = rows[ref_B];
		opt_B = CUBLAS_OP_T;
	}

    cublasStatus_t cublasError;
	cublasError = cublasSgemmBatched(handle, opt_A, opt_B,
		m, n, k,
		&alpha,
		(float const **)ptrA_dev, rows[ref_A],
		(float const **)ptrB_dev, rows[ref_B],
		&beta,
		(float **)ptrC_dev, rows[ref_C], N);
	if (cublasError != CUBLAS_STATUS_SUCCESS)
	{ 
		status = false;
		goto CLEANUP; 
	}

CLEANUP:
	// free host pointers
	if (ptrA) free(ptrA);
	if (ptrB) free(ptrB);
	if (ptrC) free(ptrC);

	// free device pointers 
	if (ptrA_dev) cudaFree(ptrA_dev);
	if (ptrB_dev) cudaFree(ptrB_dev);
	if (ptrC_dev) cudaFree(ptrC_dev);

	return status;
}


SphericalConvolution::SphericalConvolution(unsigned int const *map,
	unsigned int Fin,
	unsigned int Fout,
	unsigned int Nin,
	unsigned int Nout,
	unsigned int Nfilt)
	:map(map), Fin(Fin), Fout(Fout), Nin(Nin), Nout(Nout), Nfilt(Nfilt)
{} // constructor


SphericalConvolution::~SphericalConvolution()
{} // destructor


  // -----------------------------------------------------------------------
  //                                                                 Forward
  // -----------------------------------------------------------------------
  /*
  
  matlab arrays are column-major::
  input:  4D gpu/cpu tensor of Size (1, 1, Fin, Nin);
  filter: 4D gpu/cpu tensor of Size (1, Fin, Fout, Nfilt);
  bias:   row vector on gpu/cpu of Size (1, Fout);
  output: 4D gpu/cpu tensor of Size (1, 1, Fout, Nout);        --------TO BE COMPUTED
  map:    2D cpu array of Size (4, Nin), 0-based indexing for childID, filterID and parentID;

  Nin is equal to the number of children nodes;
  Nout is equal to the number of parent nodes;
  Nin > Nout.

  */

void SphericalConvolution::forward(float *output,
	float const *input,
	float const *filter,
	float const *bias)
{
	assert(output);
	assert(input);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle;
	cudaStream_t *streams = NULL;

	SUCCESS = true;
	cublasCreate(&handle);
	streams = (cudaStream_t *)malloc(maxStreams * sizeof(cudaStream_t));
	createStreams(streams, maxStreams);

	float *allOnesMemory = NULL, *tempOutput = NULL, alpha, beta;
	cudaError = cudaMalloc((void **)&allOnesMemory, Nin * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nin, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)allOnesMemory, Nin);

	if (filter)
	{
		// allocate memory for temp output on device， whose size is Fout x Nin
		cudaError = cudaMalloc((void **)&tempOutput, Fout * Nin * sizeof(float));
		if (cudaError != cudaSuccess) { goto done; }

		// compute temp output with cublasXgemmBatched()
		cublasSetStream(handle, streams[1]);
		SUCCESS = gemmBatched(handle,
			filter, 1, 't', input, 0, 'n', tempOutput, 2, NULL,
			map, Fin, Fout, Nin, Nout, Nfilt);
		if (!SUCCESS) { goto done; }

		cudaError = cudaDeviceSynchronize(); 
		if (cudaError != cudaSuccess) { goto done; }

		// parent-wise memcpy or summation
		beta = 0;
		int count = 0, iter = 0;
		bool prevSingle = false;
		for (int node = 0; node < Nin; ++node)
		{
			if ((node == 0) || (map[node * 4 - 2] != map[node * 4 + 2]))
			{
				unsigned int convSize = map[node * 4 + 3];

				if (convSize == 1)
				{
					prevSingle = true;
					count++;
				}
				else
				{
					ptrdiff_t parentID = map[node * 4 + 2];
					ptrdiff_t outputOffset = Fout * parentID;
					ptrdiff_t tempOutputOffset = Fout * node;

					if (prevSingle)
					{
						ptrdiff_t startOffset = count * Fout;
						cudaError = cudaMemcpyAsync((float *)output + outputOffset - startOffset,
							(float const *)tempOutput + tempOutputOffset - startOffset,
							Fout * count * sizeof(float), cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
						if (cudaError != cudaSuccess) { goto done; }
					}

					alpha = float(1) / float(convSize);

					cublasSetStream(handle, streams[iter % maxStreams]);
					cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
						Fout, 1, convSize,
						&alpha,
						(float const *)tempOutput + tempOutputOffset, Fout,
						(float const *)allOnesMemory, 1,
						&beta,
						(float *)output + outputOffset, Fout);
					if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }

					prevSingle = false;
					count = 0;

					iter++;
				}
			}
		}
		if (prevSingle)  // copy if for-loop end with convSize=1 nodes
		{
			iter++;

			int node = Nin - 1;
			ptrdiff_t parentID = map[node * 4 + 2];
			ptrdiff_t outputOffset = Fout * parentID;
			ptrdiff_t tempOutputOffset = Fout * node;

			ptrdiff_t startOffset = (count - 1) * Fout;
			cudaError = cudaMemcpyAsync((float *)output + outputOffset - startOffset,
				(float const *)tempOutput + tempOutputOffset - startOffset,
				Fout * count * sizeof(float), cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
			if (cudaError != cudaSuccess) { goto done; }
		}
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }


	if (bias)
	{
		alpha = 1;
		beta = 1;
		cublasSetStream(handle, streams[0]);
		cublasError = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			Fout, Nout, 1,
			&alpha,
			(float const *)bias, 1,
			(float const *)allOnesMemory, 1,
			&beta,
			(float *)output, Fout);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	cudaError = cudaStreamSynchronize(streams[0]);
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { SUCCESS = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);
	if (tempOutput) cudaFree(tempOutput);
	if (streams) destroyStreams(streams, maxStreams);
	cublasDestroy(handle);
}


// -----------------------------------------------------------------------
//                                                                Backward
// -----------------------------------------------------------------------
/*

matlab arrays are column-major::
input:     4D gpu tensor of Size (1, 1, Fin, Nin);
filter:    4D gpu tensor of Size (1, Fin, Fout, Nfilt);
derOutput: 4D gpu tensor of Size (1, 1, Fout, Nout);
map:       2D cpu array of Size (4, Nin), 0-based indexing for childID, filterID and parentID;
derInput:  4D gpu tensor of Size (1, 1, Fin, Nin);             --------TO BE COMPUTED
derFilter: 4D gpu tensor of Size (1, Fin, Fout, Nfilt);  --------TO BE COMPUTED
derBias:   row vector on gpu of Size (1, Fout);						--------TO BE COMPUTED

Nin is equal to the number of children nodes;
Nout is equal to the number of parent nodes;
Nin > Nout.

*/

void SphericalConvolution::backward(float *derInput,
	float *derFilter,
	float *derBias,
	float const *input,
	float const *filter,
	float const *derOutput)
{
	// for all derivatives
	assert(derOutput);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle;
	cudaStream_t *streams = NULL;

	SUCCESS = true;
	cublasCreate(&handle);
	streams = (cudaStream_t *)malloc(maxStreams * sizeof(cudaStream_t));
	createStreams(streams, maxStreams);

	// variables declaration
	float *allOnesMemory = NULL, *avgDerOutput = NULL, *tempDerFilter = NULL, alpha;
	float const beta = 0; // constant setting
	int *filtPerNum = NULL;

	// allocate memory to allOnesMemory and set it to ones
	cudaError = cudaMalloc((void **)&allOnesMemory, Nin * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nin, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)allOnesMemory, Nin);
	setToZeros << < divideAndRoundUp(Fin * Fout * Nfilt, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)derFilter, Fin * Fout * Nfilt);

	// allocate memory to avgDerOutput and initialize it with derOutput
	cudaError = cudaMalloc((void **)&avgDerOutput, Fout * Nout * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	cudaError = cudaMemcpyAsync(avgDerOutput, derOutput,
		Fout * Nout * sizeof(float), cudaMemcpyDeviceToDevice, streams[1]);
	if (cudaError != cudaSuccess) { goto done; }

	cudaError = cudaStreamSynchronize(streams[1]);
	if (cudaError != cudaSuccess) { goto done; }

	// average each feature in derOutput with its corresponding convSize
	int iter;
    iter = 0;
	for (int node = 0; node < Nin; ++node)
	{
		if ((node == 0) || (map[node * 4 - 2] != map[node * 4 + 2]))
		{
			unsigned int convSize = map[node * 4 + 3];

			if (convSize > 1)
			{
				ptrdiff_t parentID = map[node * 4 + 2];
				ptrdiff_t outputOffset = Fout * parentID;

				alpha = float(1) / float(convSize);

				cublasSetStream(handle, streams[iter % maxStreams]);
				cublasError = cublasSscal(handle, Fout,
					&alpha,
					(float *)avgDerOutput + outputOffset, 1);
				if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }

				iter++;
			}
		}
	}

	if (derBias)
	{
		alpha = 1;
		cublasSetStream(handle, streams[0]);
		cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
			1, Fout, Nout,
			&alpha,
			(float const *)allOnesMemory, 1,
			(float const *)derOutput, Fout,
			&beta,
			(float *)derBias, 1);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

	if (derInput)
	{
		assert(filter);

		cublasSetStream(handle, streams[0]);
		SUCCESS = gemmBatched(handle,
			filter, 1, 'n', avgDerOutput, 2, 'n', derInput, 0, NULL,
			map, Fin, Fout, Nin, Nout, Nfilt);
		if (!SUCCESS) { goto done; }
	}

	if (derFilter)
	{
		assert(input);

		ptrdiff_t filterVolume = Fout * Fin;

		cudaError = cudaMalloc((void **)&tempDerFilter, filterVolume * Nin * sizeof(float));
		if (cudaError != cudaSuccess) { goto done; }
		filtPerNum = (int *)malloc(Nfilt * sizeof(int));

		cublasSetStream(handle, streams[1]);
		SUCCESS = gemmBatched(handle,
			input, 0, 'n', avgDerOutput, 2, 't', tempDerFilter, 1, filtPerNum,
			map, Fin, Fout, Nin, Nout, Nfilt);
		if (!SUCCESS) { goto done; }

		cudaError = cudaStreamSynchronize(streams[1]);
		if (cudaError != cudaSuccess) { goto done; }

		alpha = 1;
		int iter = 0, count = 0;
		for (int idx = 0; idx < Nfilt; idx++)
		{
			if (filtPerNum[idx] == 1)
			{
				ptrdiff_t filterGrpOffset = filterVolume * idx;
				ptrdiff_t tempFilterGrpOffset = filterVolume * count;

				cudaError = cudaMemcpyAsync((float *)derFilter + filterGrpOffset,
					(float const *)tempDerFilter + tempFilterGrpOffset,
					filterVolume * sizeof(float),
					cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
				if (cudaError != cudaSuccess) { goto done; }				

				count += filtPerNum[idx];
				iter++;
				continue;
			}
			if (filtPerNum[idx] > 1)
			{
				ptrdiff_t filterGrpOffset = filterVolume * idx;
				ptrdiff_t tempFilterGrpOffset = filterVolume * count;

				cublasSetStream(handle, streams[iter % maxStreams]);
				cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
					filterVolume, 1, filtPerNum[idx],
					&alpha,
					(float const *)tempDerFilter + tempFilterGrpOffset, filterVolume,
					(float const *)allOnesMemory, 1,
					&beta,
					(float *)derFilter + filterGrpOffset, filterVolume);
				if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }

				count += filtPerNum[idx];
				iter++;
				continue;
			}
		}
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { SUCCESS = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);
	if (avgDerOutput) cudaFree(avgDerOutput);
	if (tempDerFilter) cudaFree(tempDerFilter);
	if (filtPerNum) free(filtPerNum);
	if (streams) destroyStreams(streams, maxStreams);
	cublasDestroy(handle);
}



SphericalDeconvolution::SphericalDeconvolution(unsigned int const *map,
	unsigned int Fin,
	unsigned int Fout,
	unsigned int Nin,
	unsigned int Nout,
	unsigned int Nfilt)
	:map(map), Fin(Fin), Fout(Fout), Nin(Nin), Nout(Nout), Nfilt(Nfilt)
{} // constructor


SphericalDeconvolution::~SphericalDeconvolution()
{} // destructor


  // -----------------------------------------------------------------------
  //										 Spherical Deconvolution Forward
  // -----------------------------------------------------------------------
  /*
  
  matlab arrays are column-major::
  input:  4D gpu/cpu tensor of Size (1, 1, Fin, Nin);
  filter: 4D gpu/cpu tensor of Size (1, Fin, Fout, Nfilt);
  bias:   row vector on gpu/cpu of Size (1, Fout);
  output: 4D gpu/cpu tensor of Size (1, 1, Fout, Nout);        --------TO BE COMPUTED
  map:    2D cpu array of Size (4, Nin), 0-based indexing for childID, filterID and parentID;

  Nin is equal to the number of children nodes;
  Nout is equal to the number of parent nodes;
  Nin < Nout.

  */

void SphericalDeconvolution::forward(float *output,
	float const *input,
	float const *filter,
	float const *bias)
{
	assert(output);
	assert(input);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle; // only the default stream is used 

	SUCCESS = true;
	cublasCreate(&handle);

	float *allOnesMemory = NULL;
	float const alpha = 1, beta = 1;
	cudaError = cudaMalloc((void **)&allOnesMemory, Nout * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nout, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >> >
		((float *)allOnesMemory, Nout);

	if (filter)
	{
		SUCCESS = gemmBatched(handle,
			filter, 1, 't', input, 0, 'n', output, 2, NULL,
			map, Fin, Fout, Nin, Nout, Nfilt);
		if (!SUCCESS) { goto done; }
	}	

	if (bias)
	{
		cublasError = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			Fout, Nout, 1,
			&alpha,
			(float const *)bias, 1,
			(float const *)allOnesMemory, 1,
			&beta,
			(float *)output, Fout);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { SUCCESS = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);	
	cublasDestroy(handle);
}


// -----------------------------------------------------------------------
//                                        Spherical Deconvolution Backward
// -----------------------------------------------------------------------
/*

matlab arrays are column-major::
input:     4D gpu tensor of Size (1, 1, Fin, Nin);
filter:    4D gpu tensor of Size (1, Fin, Fout, Nfilt);
derOutput: 4D gpu tensor of Size (1, 1, Fout, Nout);
map:       2D cpu array of Size (4, Nin), 0-based indexing for childID, filterID and parentID;
derInput:  4D gpu tensor of Size (1, 1, Fin, Nin);             --------TO BE COMPUTED
derFilter: 4D gpu tensor of Size (1, Fin, Fout, Nfilt);  --------TO BE COMPUTED
derBias:   row vector on gpu of Size (1, Fout);						--------TO BE COMPUTED

Nin is equal to the number of children nodes;
Nout is equal to the number of parent nodes;
Nin < Nout.

*/

void SphericalDeconvolution::backward(float *derInput,
	float *derFilter,
	float *derBias,
	float const *input,
	float const *filter,
	float const *derOutput)
{
	// for all derivatives
	assert(derOutput);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle;
	cudaStream_t *streams = NULL;

	SUCCESS = true;
	cublasCreate(&handle);
	streams = (cudaStream_t *)malloc(maxStreams * sizeof(cudaStream_t));
	createStreams(streams, maxStreams);

	float *allOnesMemory = NULL, *tempDerInput = NULL, *tempDerFilter = NULL;
	float const alpha = 1, beta = 0;
	int *filtPerNum = NULL;

	cudaError = cudaMalloc((void **)&allOnesMemory, Nout * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nout, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)allOnesMemory, Nout);
	setToZeros << < divideAndRoundUp(Fin * Fout * Nfilt, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)derFilter, Fin * Fout * Nfilt);

	if (derBias)
	{
		cublasSetStream(handle, streams[0]);
		cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
			1, Fout, Nout,
			&alpha,
			(float const*)allOnesMemory, 1,
			(float const*)derOutput, Fout,
			&beta,
			(float *)derBias, 1);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	if (derInput)
	{
		assert(filter);

		cudaError = cudaMalloc((void **)&tempDerInput, Fin * Nout * sizeof(float));
		if (cudaError != cudaSuccess) { goto done; }

		cublasSetStream(handle, streams[1]);
		SUCCESS = gemmBatched(handle,
			filter, 1, 'n', derOutput, 2, 'n', tempDerInput, 0, NULL,
			map, Fin, Fout, Nin, Nout, Nfilt);
		if (!SUCCESS) { goto done; }

		cudaError = cudaStreamSynchronize(streams[1]);
		if (cudaError != cudaSuccess) { goto done; }

		int count = 0, iter = 0;
		bool prevSingle = false;
		for (int node = 0; node < Nout; ++node)
		{
			if ((node == 0) || (map[node * 4 - 4] != map[node * 4]))
			{
				unsigned int convSize = map[node * 4 + 3];

				if (convSize == 1)
				{
					prevSingle = true;
					count++;
				}
				else
				{
					ptrdiff_t dataOffset = Fin * map[node * 4];
					ptrdiff_t tempDataOffset = Fin * node;

					if (prevSingle)
					{
						ptrdiff_t startOffset = count * Fin;
						cudaError = cudaMemcpyAsync((float *)derInput + dataOffset - startOffset,
							(float const *)tempDerInput + tempDataOffset - startOffset,
							Fin * count * sizeof(float), cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
						if (cudaError != cudaSuccess) { goto done; }
					}

					cublasSetStream(handle, streams[iter % maxStreams]);
					cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
						Fin, 1, convSize,
						&alpha,
						(float const *)tempDerInput + tempDataOffset, Fin,
						(float const *)allOnesMemory, 1,
						&beta,
						(float *)derInput + dataOffset, Fin);
					if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }

					prevSingle = false;
					count = 0;

					iter++;
				}
			}
		}
		if (prevSingle)  // copy if for-loop end with convSize=1 nodes
		{
			iter++;

			int node = Nout - 1;
			ptrdiff_t dataOffset = Fin * map[node * 4];
			ptrdiff_t tempDataOffset = Fin * node;

			ptrdiff_t startOffset = (count - 1) * Fin;
			cudaError = cudaMemcpyAsync((float *)derInput + dataOffset - startOffset,
				(float const *)tempDerInput + tempDataOffset - startOffset,
				Fin * count * sizeof(float), cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
			if (cudaError != cudaSuccess) { goto done; }
		}
	}

	if (derFilter)
	{
		assert(input);

		ptrdiff_t filterVolume = Fout * Fin;

		cudaError = cudaMalloc((void **)&tempDerFilter, filterVolume * Nout * sizeof(float));
		if (cudaError != cudaSuccess) { goto done; }
		filtPerNum = (int *)malloc(Nfilt * sizeof(int));

		cublasSetStream(handle, streams[2]);
		SUCCESS = gemmBatched(handle,
			input, 0, 'n', derOutput, 2, 't', tempDerFilter, 1, filtPerNum,
			map, Fin, Fout, Nin, Nout, Nfilt);
		if (!SUCCESS) { goto done; }

		cudaError = cudaStreamSynchronize(streams[2]);
		if (cudaError != cudaSuccess) { goto done; }

		int iter = 0, count = 0;
		for (int idx = 0; idx < Nfilt; idx++)
		{
			if (filtPerNum[idx] == 1)
			{
				ptrdiff_t filterGrpOffset = filterVolume * idx;
				ptrdiff_t tempFilterGrpOffset = filterVolume * count;

				cudaError = cudaMemcpyAsync((float *)derFilter + filterGrpOffset,
					(float const *)tempDerFilter + tempFilterGrpOffset,
					filterVolume * sizeof(float),
					cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
				if (cudaError != cudaSuccess) { goto done; }

				count += filtPerNum[idx];
				iter++;
				continue;
			}
			if (filtPerNum[idx] > 1)
			{
				ptrdiff_t filterGrpOffset = filterVolume * idx;
				ptrdiff_t tempFilterGrpOffset = filterVolume * count;

				cublasSetStream(handle, streams[iter % maxStreams]);
				cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
					filterVolume, 1, filtPerNum[idx],
					&alpha,
					(float const *)tempDerFilter + tempFilterGrpOffset, filterVolume,
					(float const *)allOnesMemory, 1,
					&beta,
					(float *)derFilter + filterGrpOffset, filterVolume);
				if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }

				count += filtPerNum[idx];
				iter++;
				continue;
			}
		}
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { SUCCESS = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);
	if (tempDerInput) cudaFree(tempDerInput);
	if (tempDerFilter) cudaFree(tempDerFilter);
	if (filtPerNum) free(filtPerNum);	
	if (streams) destroyStreams(streams, maxStreams);
	cublasDestroy(handle);
}



FullyConnected::FullyConnected(unsigned int const *map,
	unsigned int Fin,
	unsigned int Fout,
	unsigned int Nin,
	unsigned int Nout,
	unsigned int Nfilt)
	:map(map), Fin(Fin), Fout(Fout), Nin(Nin), Nout(Nout), Nfilt(Nfilt)
{} // constructor


FullyConnected::~FullyConnected()
{} // destructor

// -----------------------------------------------------------------------
//										 Spherical Deconvolution Forward
// -----------------------------------------------------------------------
/*

matlab arrays are column-major::
input:  4D gpu/cpu tensor of Size (1, 1, Fin, Nin);
filter: 4D gpu/cpu tensor of Size (1, Fin, Fout, Nfilt);
bias:   row vector on gpu/cpu of Size (1, Fout);
output: 4D gpu/cpu tensor of Size (1, 1, Fout, Nout);        --------TO BE COMPUTED
map:    2D cpu array of Size (4, Nin), 0-based indexing for childID, filterID and parentID;

Nin is equal to the number of children nodes;
Nout is equal to the number of parent nodes;
!!! Nin = Nout.

*/

void FullyConnected::forward(float *output,
	float const *input,
	float const *filter,
	float const *bias)
{
	assert(output);
	assert(input);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle;

	SUCCESS = true;
	cublasCreate(&handle);

	float *allOnesMemory = NULL, beta;
	float const alpha = 1; // constant setting
	cudaError = cudaMalloc((void **)&allOnesMemory, Nin * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nin, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >> >
		((float *)allOnesMemory, Nin);

	if (filter)
	{
		beta = 0;
		cublasError = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			Fout, Nout, Fin,
			&alpha,
			(float const *)filter, Fin,
			(float const *)input, Fin,
			&beta,
			(float *)output, Fout);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	if (bias)
	{
		beta = 1;
		cublasError = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			Fout, Nout, 1,
			&alpha,
			(float const *)bias, 1,
			(float const *)allOnesMemory, 1,
			&beta,
			(float *)output, Fout);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { SUCCESS = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);
	cublasDestroy(handle);
}


// -----------------------------------------------------------------------
//												  Fully Connected Backward
// -----------------------------------------------------------------------
/*

matlab arrays are column-major::
input:     4D gpu tensor of Size (1, 1, Fin, Nin);
filter:    4D gpu tensor of Size (1, Fin, Fout, Nfilt);
derOutput: 4D gpu tensor of Size (1, 1, Fout, Nout);
map:       2D cpu array of Size (4, Nin), 0-based indexing for childID, filterID and parentID;
derInput:  4D gpu tensor of Size (1, 1, Fin, Nin);             --------TO BE COMPUTED
derFilter: 4D gpu tensor of Size (1, Fin, Fout, Nfilt);  --------TO BE COMPUTED
derBias:   row vector on gpu of Size (1, Fout);						--------TO BE COMPUTED

Nin is equal to the number of children nodes;
Nout is equal to the number of parent nodes;
!!! Nin = Nout.

*/
void FullyConnected::backward(float *derInput,
	float *derFilter,
	float *derBias,
	float const *input,
	float const *filter,
	float const *derOutput)
{
	// for all derivatives
	assert(derOutput);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle;
	cudaStream_t *streams = NULL;

	SUCCESS = true;
	cublasCreate(&handle);
	streams = (cudaStream_t *)malloc(maxStreams * sizeof(cudaStream_t));
	createStreams(streams, maxStreams);

	// variables declaration
	float *allOnesMemory = NULL;
	float const alpha = 1, beta = 0; // constant setting

	// allocate memory to allOnesMemory and set it to ones
	cudaError = cudaMalloc((void **)&allOnesMemory, Nin * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nin, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)allOnesMemory, Nin);
	setToZeros << < divideAndRoundUp(Fin * Fout * Nfilt, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[2] >> >
		((float *)derFilter, Fin * Fout * Nfilt);

	if (derBias)
	{
		cublasSetStream(handle, streams[0]);
		cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
			1, Fout, Nout,
			&alpha,
			(float const *)allOnesMemory, 1,
			(float const *)derOutput, Fout,
			&beta,
			(float *)derBias, 1);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	if (derInput)
	{
		assert(filter);
		cublasSetStream(handle, streams[1]);
		cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			Fin, Nin, Fout,
			&alpha,
			(float const *)filter, Fin,
			(float const *)derOutput, Fout,
			&beta,
			(float *)derInput, Fin);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	if (derFilter)
	{
		assert(input);

		ptrdiff_t filterVolume = Fout * Fin;

		cublasSetStream(handle, streams[2]);
		cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
			Fin, Fout, Nin,
			&alpha,
			(float const *)input, Fin,
			(float const *)derOutput, Fout,
			&beta,
			(float *)derFilter, Fin);
		if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { SUCCESS = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);
	if (streams) destroyStreams(streams, maxStreams);
	cublasDestroy(handle);
}