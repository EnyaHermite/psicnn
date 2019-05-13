# psicnn
Octree guided cnn with spherical kernels for 3D point clouds

We use matconvnet in out experiment and hence implement the spherical kernel in a matconvnet style. The file getOctreeBatch.m 
shows how we organize simple batch point clouds after octree construction. 

For ubuntu system, use the file mexSources_ubuntu.m to compile the source files.  We have compiled the files successfully under the configuration Matlab2017b+CUDA8.0+gcc4.8.  
For windows system, use mexSources_windows.m instead. Our configuration is Matlab2017b+CUDA8.0+VS2015.

The file demo.m shows how to prepare batch octree inputs, and use the spherical kernel to extract features. The files in the folder debug_helper is provided to make sure that the compiled kernel is working properly.


 
If you find our work useful in your research, please consider citing:  
@article{lei2019octree,  
      title={Octree guided CNN with Spherical Kernels for 3D Point Clouds},  
      author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},  
      journal={IEEE Conference on Computer Vision and Pattern Recognition},  
      year={2019}  
}
