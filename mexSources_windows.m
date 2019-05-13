warning off;

% mex octree construction codes
mex mexOctreeMap.cpp Octree.cpp

% mex kernel of spherical convolution and other cuda-based codes
setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0');
setenv('CUDA_BIN_PATH','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin');
setenv('CUDA_LIB_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib');

mexcuda -v -lcublas.lib mexSphericalConvolution.cu nnsphconv.cu
mexcuda -v -lcublas.lib mexFastSum.cu
