warning('off','all')

% mex octree construction codes
mex mexOctreeMap.cpp Octree.cpp

% mex kernel of spherical convolution and other cuda-based codes
setenv('CUDA_PATH', '/usr/local/cuda');
setenv('CUDA_BIN_PATH','/usr/local/cuda/bin');
setenv('CUDA_LIB_PATH', '/usr/local/cuda/lib64');

mexcuda -v '-I/usr/local/cuda/include' '-L/usr/local/cuda/lib64' -lcublas mexSphericalConvolution.cu nnsphconv.cu
