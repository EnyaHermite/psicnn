# psicnn
Octree guided cnn with spherical kernels for 3D point clouds

We use matconvnet in out experiment and hence implement the spherical kernel in a matconvnet style. The file getOctreeBatch.m 
shows how we organize simple batch point clouds after octree construction. 

For ubuntu system, use the file mexSources_ubuntu.m to compile the source files. For windows system, use mexSources_windows.m instead. 


Citation
If you find our work useful in your research, please consider citing:

@article{lei2019octree,  
      title={Octree guided CNN with Spherical Kernels for 3D Point Clouds},  
      author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},  
      journal={IEEE Conference on Computer Vision and Pattern Recognition},  
      year={2019}  
}
