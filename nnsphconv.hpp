// @author Huan LEI
// All rights reserved.

#ifndef __nnsphconv_hpp__
#define __nnsphconv_hpp__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

class SphericalConvolution {
public:
	SphericalConvolution(unsigned int const *map,
		unsigned int Fin,
		unsigned int Fout,
		unsigned int Nin,
		unsigned int Nout,
		unsigned int Nfilt);

	void forward(float *output,
		float const *input,
		float const *filter,
		float const *bias);

	void backward(float *derInput,
		float *derFilter,
		float *derBias,
		float const *input,
		float const *filter,
		float const *derOutput);

	~SphericalConvolution();	
	
	unsigned int Fin, Fout, Nin, Nout, Nfilt;
	unsigned int const *map; /* size = (4, inputSize), one column = {childID, filterID, parentID, convSize}	*/
	bool SUCCESS = false; /* indicate whether the forward/backward computation is successful or not */
};


class SphericalDeconvolution {
public:
	SphericalDeconvolution(unsigned int const *map,
		unsigned int Fin,
		unsigned int Fout,
		unsigned int Nin,
		unsigned int Nout,
		unsigned int Nfilt);

	void forward(float *output,
		float const *input,
		float const *filter,
		float const *bias);

	void backward(float *derInput,
		float *derFilter,
		float *derBias,
		float const *input,
		float const *filter,
		float const *derOutput);

	~SphericalDeconvolution();	
		
	unsigned int Fin, Fout, Nin, Nout, Nfilt;
	unsigned int const *map; /* size = (4, inputSize), one column = {childID, filterID, parentID, convSize}	*/
	bool SUCCESS = false; /* indicate whether the forward/backward computation is successful or not */
};


class FullyConnected {
public:
	FullyConnected(unsigned int const *map,
		unsigned int Fin,
		unsigned int Fout,
		unsigned int Nin,
		unsigned int Nout,
		unsigned int Nfilt);

	void forward(float *output,
		float const *input,
		float const *filter,
		float const *bias);

	void backward(float *derInput,
		float *derFilter,
		float *derBias,
		float const *input,
		float const *filter,
		float const *derOutput);

	~FullyConnected();

	unsigned int Fin, Fout, Nin, Nout, Nfilt;
	unsigned int const *map; /* size = (4, N), one column = {id, 0, id, 1}	*/
	bool SUCCESS = false; /* indicate whether the forward/backward computation is successful or not */
};
#endif /* defined(__nnsphconv_hpp__) */