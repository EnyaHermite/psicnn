#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cassert>
#include <stdlib.h>
#include <cuda.h>
#include <device_functions.h>

#define VL_CUDA_NUM_THREADS 1024
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


bool speedup_sum(float *output,
	float const *input,
	unsigned int const *nodeSize,
	unsigned int const Nin,
	unsigned int const Nout,
	unsigned int const Fin)
{
	assert(output);
	assert(input);

	cudaError_t cudaError;
	cublasStatus_t cublasError;
	cublasHandle_t handle;
	cudaStream_t *streams = NULL;

	bool status = true;
	cublasCreate(&handle);
	streams = (cudaStream_t *)malloc(maxStreams * sizeof(cudaStream_t));
	createStreams(streams, maxStreams);

	float *allOnesMemory = NULL;
	cudaError = cudaMalloc((void **)&allOnesMemory, Nin * sizeof(float));
	if (cudaError != cudaSuccess) { goto done; }
	setToOnes << < divideAndRoundUp(Nin, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS, 0, streams[0] >> >
		((float *)allOnesMemory, Nin);

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

	// parent-wise memcpy or summation
	float  alpha, beta;
	alpha = 1;
	beta = 0;
	int count;
	count = 0;
	for (int iter = 0; iter < Nout; ++iter)
	{
		unsigned int convSize = nodeSize[iter];
		ptrdiff_t inputOffset = Fin * count;
		ptrdiff_t outputOffset = Fin * iter;

		if (convSize == 1) // copy if for-loop end with convSize=1 nodes
		{
			cudaError = cudaMemcpyAsync((float *)output + outputOffset,
				(float const *)input + inputOffset,
				Fin * sizeof(float), cudaMemcpyDeviceToDevice, streams[iter % maxStreams]);
			if (cudaError != cudaSuccess) { goto done; }
		}
		else
		{
			cublasSetStream(handle, streams[iter % maxStreams]);
			cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
				Fin, 1, convSize,
				&alpha,
				(float const *)input + inputOffset, Fin,
				(float const *)allOnesMemory, 1,
				&beta,
				(float *)output + outputOffset, Fin);
			if (cublasError != CUBLAS_STATUS_SUCCESS) { goto done; }
		}

		count += convSize;
	}

	cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess) { goto done; }

done:
	if (cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS) { status = false; }
	if (allOnesMemory) cudaFree(allOnesMemory);
	if (streams) destroyStreams(streams, maxStreams);
	cublasDestroy(handle);

	return status;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	mxArray const *prhs[])
{
	float const  *input;
	float *output;
	unsigned int Fin, Nin, Nout;
	unsigned int const *nodeSize;
	bool err = false;
	char *errMsg;

	mxGPUArray const *dev_input;
	mxGPUArray *dev_output;
	mwSize const *dims;
	mwSize dims_4D[] = { 1, 1, 1, 1 };
	mwSize ndim;

	mxInitGPU(); /* Initialize the MathWorks GPU API. */

				 /* -------------------------------------------------------------- */
				 /*										Check the input arguments */
				 /* -------------------------------------------------------------- */

	if (nrhs < 2) { mexErrMsgTxt("Not enough input arguments."); }
	if (nrhs > 2) { mexErrMsgTxt("Too many inputs."); }

	// get pointer to gpuArray input
	dev_input = mxGPUCreateFromMxArray(prhs[0]);
	if (mxGPUGetClassID(dev_input) == mxSINGLE_CLASS)
	{
		input = (float const *)mxGPUGetDataReadOnly(dev_input);
	}
	else
	{
		mexErrMsgTxt("Only single format of network input is supported.");
	}

	// get pointer to cpuArray map
	if (mxGetClassID(prhs[1]) == mxUINT32_CLASS)
	{
		nodeSize = (unsigned int const *)mxGetData(prhs[1]);
		Nout = mxGetNumberOfElements(prhs[1]);
	}
	else
	{
		mexErrMsgTxt("Only unsigned int format of nodeSize is supported.");
	}

	// parse the network input
	ndim = mxGPUGetNumberOfDimensions(dev_input);
	dims = mxGPUGetDimensions(dev_input);
	if (ndim<3 || ndim>4)
	{
		mexErrMsgTxt("The network input must be a 4D matrix.");
	}
	else
	{
		Fin = dims[0] * dims[1] * dims[2];
		Nin = (ndim == 3) ? 1 : dims[3];
	}

	/* -------------------------------------------------------------- */
	/*													   Do the work*/
	/* -------------------------------------------------------------- */
	dims_4D[2] = Fin; dims_4D[3] = Nout;
	dev_output = mxGPUCreateGPUArray(4, dims_4D, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	output = (float *)mxGPUGetData(dev_output);
	bool status = speedup_sum(output, input, nodeSize, Nin, Nout, Fin);

	if (status)
	{
		plhs[0] = mxGPUCreateMxArrayOnGPU(dev_output);
	}
	else
	{
		err = true;
		errMsg = "The speed-up summation is failed.";
	}

	mxGPUDestroyGPUArray(dev_input);
	mxGPUDestroyGPUArray(dev_output);
	if (err) mexErrMsgTxt(errMsg);
}

