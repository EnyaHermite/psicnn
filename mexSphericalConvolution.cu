#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "nnsphconv.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	mxArray const *prhs[])
{
	float const  *input, *filter, *bias, *derOutput;
	float *output, *derInput, *derFilter, *derBias;
	unsigned int Fin, Fout, Nin, Nout, Nfilt;
	unsigned int const *map;
	bool backMode, err = false;
	char *errMsg;

	mxGPUArray const *dev_input, *dev_filter, *dev_bias, *dev_derOutput;
	mxGPUArray *dev_output, *dev_derInput, *dev_derFilter, *dev_derBias;
	mwSize const *dims;
	mwSize dims_4D[] = { 1, 1, 1, 1 }, dims_2D[] = { 1, 1 };
	mwSize ndim;

	mxInitGPU(); /* Initialize the MathWorks GPU API. */

	/* -------------------------------------------------------------- */
	/*										Check the input arguments */
	/* -------------------------------------------------------------- */

	if (nrhs < 4) { mexErrMsgTxt("Not enough input arguments."); }
	if (nrhs > 5) { mexErrMsgTxt("Too many inputs."); }
	if (nrhs == 4) { backMode = false; }
	if (nrhs == 5) { backMode = true; }

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

	// get pointer to gpuArray filter
	dev_filter = mxGPUCreateFromMxArray(prhs[1]);
	if (mxGPUGetClassID(dev_filter) == mxSINGLE_CLASS)
	{
		filter = (float const *)mxGPUGetDataReadOnly(dev_filter);
	}
	else
	{
		mexErrMsgTxt("Only single format of network filter is supported.");
	}

	// get pointer to gpuArray bias
	if (!backMode)
	{
		dev_bias = mxGPUCreateFromMxArray(prhs[2]);
		if (mxGPUGetClassID(dev_bias) == mxSINGLE_CLASS)
		{
			bias = (float const *)mxGPUGetDataReadOnly(dev_bias);
		}

		else
		{
			mexErrMsgTxt("Only single format of network bias is supported.");
		}
	}

	// get pointer to cpuArray map
	if (mxGetClassID(prhs[3]) == mxUINT32_CLASS)
	{
		map = (unsigned int const *)mxGetData(prhs[3]);
	}
	else
	{
		mexErrMsgTxt("Only unsigned int format of network map is supported.");
	}

	// get pointer to gpuArray output derivative
	if (backMode)
	{
		dev_derOutput = mxGPUCreateFromMxArray(prhs[4]);
		if (mxGPUGetClassID(dev_derOutput) == mxSINGLE_CLASS)
		{
			derOutput = (float const *)mxGPUGetDataReadOnly(dev_derOutput);
		}
		else
		{
			mexErrMsgTxt("Only single format of network output derivative is supported.");
		}
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

	// parse the network filter
	ndim = mxGPUGetNumberOfDimensions(dev_filter);
	dims = mxGPUGetDimensions(dev_filter);
	if (ndim<3 || ndim>4)
	{
		mexErrMsgTxt("The network filter must be a 4D matrix.");
	}
	else
	{
		if ((dims[0] * dims[1]) != Fin)
		{
			mexErrMsgTxt("The input feature size of the network filter must equal " \
				"that of the network input.");
		}
		Fout = dims[2];
		Nfilt = (ndim == 3) ? 1 : dims[3];
	}

	// parse the network bias
	if (!backMode)
	{
		if (mxGPUGetNumberOfElements(dev_bias) != Fout)
		{
			mexErrMsgTxt("Number of elements in network bias must equal the output feature size.");
		}
	}

	// parse map from the network input to the network output
	if (mxGetM(prhs[3]) != 4)
	{
		mexErrMsgTxt("The first dimension of map must be 4.");
	}
	int N = mxGetNumberOfElements(prhs[3]) / 4; // here we don't force map to be a 2D matrix
	Nout = map[N * 4 - 2] + 1;
	if (N != Nin & N != Nout) 
	{
		mexErrMsgTxt("The second dimension of map must equal the number of network input" \
			"in spherical convolution, and the number of network output in spherical deconvolution.");
	}
	

	// parse the network output derivative
	if (backMode)
	{
		ndim = mxGPUGetNumberOfDimensions(dev_derOutput);
		dims = mxGPUGetDimensions(dev_derOutput);
		if (ndim<3 || ndim>4)
		{
			mexErrMsgTxt("The network output derivative must be a 4D matrix.");
		}
		else
		{
			if ((dims[0] * dims[1] * dims[2]) != Fout)
			{
				mexErrMsgTxt("Feature size of the network output derivative " \
					"must equal the output feature size of the network filter.");
			}
			if ((ndim == 3 & Nout != 1) & (dims[3] != Nout))
			{
				mexErrMsgTxt("The number of the network output derivative  " \
					"must match the one given in map.");
			}
		}
	}

	/* -------------------------------------------------------------- */
	/*													   Do the work*/
	/* -------------------------------------------------------------- */
	
	if (!backMode) // forward propagation
	{
		dims_4D[2] = Fout; dims_4D[3] = Nout;
		dev_output = mxGPUCreateGPUArray(4, dims_4D, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		output = (float *)mxGPUGetData(dev_output);

		if (Nin > Nout)  // spherical convolution
		{
			SphericalConvolution op(map, Fin, Fout, Nin, Nout, Nfilt);
			op.forward(output, input, filter, bias);

			if (op.SUCCESS)
			{
				plhs[0] = mxGPUCreateMxArrayOnGPU(dev_output);
			}
			else
			{
				err = true;
				errMsg = "The forward spherical convolution is failed.";
			}
		}		
		if(Nin <= Nout) // spherical deconvolution
		{
			SphericalDeconvolution op(map, Fin, Fout, Nin, Nout, Nfilt);
			op.forward(output, input, filter, bias);

			if (op.SUCCESS)
			{
				plhs[0] = mxGPUCreateMxArrayOnGPU(dev_output);
			}
			else
			{
				err = true;
				errMsg = "The forward spherical deconvolution is failed.";
			}
		}

		mxGPUDestroyGPUArray(dev_input);
		mxGPUDestroyGPUArray(dev_filter);
		mxGPUDestroyGPUArray(dev_bias);
		mxGPUDestroyGPUArray(dev_output);
		if (err) mexErrMsgTxt(errMsg);
	}
	else
	{
		dims_4D[2] = Fin; dims_4D[3] = Nin;
		dev_derInput = mxGPUCreateGPUArray(4, dims_4D, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		derInput = (float *)mxGPUGetData(dev_derInput);

		dims_4D[1] = Fin; dims_4D[2] = Fout; dims_4D[3] = Nfilt;
		dev_derFilter = mxGPUCreateGPUArray(4, dims_4D, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		derFilter = (float *)mxGPUGetData(dev_derFilter);

		dims_2D[1] = Fout;
		dev_derBias = mxGPUCreateGPUArray(2, dims_2D, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		derBias = (float *)mxGPUGetData(dev_derBias);

		if (Nin > Nout) // spherical convolution
		{
			SphericalConvolution op(map, Fin, Fout, Nin, Nout, Nfilt);
			op.backward(derInput, derFilter, derBias, input, filter, derOutput);

			if (op.SUCCESS)
			{
				plhs[0] = mxGPUCreateMxArrayOnGPU(dev_derInput);
				plhs[1] = mxGPUCreateMxArrayOnGPU(dev_derFilter);
				plhs[2] = mxGPUCreateMxArrayOnGPU(dev_derBias);
			}
			else
			{
				err = true;
				errMsg = "The backward spherical convolution is failed.";
			}
		}		
		if (Nin <= Nout)// spherical deconvolution
		{
			SphericalDeconvolution op(map, Fin, Fout, Nin, Nout, Nfilt);
			op.backward(derInput, derFilter, derBias, input, filter, derOutput);

			if (op.SUCCESS)
			{
				plhs[0] = mxGPUCreateMxArrayOnGPU(dev_derInput);
				plhs[1] = mxGPUCreateMxArrayOnGPU(dev_derFilter);
				plhs[2] = mxGPUCreateMxArrayOnGPU(dev_derBias);
			}
			else
			{
				err = true;
				errMsg = "The backward spherical deconvolution is failed.";
			}
		}

		mxGPUDestroyGPUArray(dev_input);
		mxGPUDestroyGPUArray(dev_filter);
		mxGPUDestroyGPUArray(dev_derOutput);
		mxGPUDestroyGPUArray(dev_derInput);
		mxGPUDestroyGPUArray(dev_derFilter);
		mxGPUDestroyGPUArray(dev_derBias);
		if (err) mexErrMsgTxt(errMsg);
	}
}

