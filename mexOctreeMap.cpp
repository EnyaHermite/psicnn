#include "Octree.hpp"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const float *pointCloud; // pointer to the input point cloud ( each point (x,y,z) has no more than 20 features)
	int maxLevels, minPoints; // parameter configuration for the octree
	int size, ndim; // size and channels of the input point cloud
	string nodePointType; // specify which type of node point to compute
	bool deconv; // whether to output the deconvolution map or not

	if (nrhs != 4) {
		mexErrMsgTxt("Four inputs required.");
	}
	if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("Only single format of point cloud is supported.");
	}
	if (mxGetN(prhs[0]) < 3 || mxGetN(prhs[0]) > 23)
	{
		mexErrMsgTxt("Column size of the input point cloud should be between 3 and 23, inclusive!");
	}
	if (!mxIsScalar(prhs[1]) || !mxIsScalar(prhs[2]))
	{
		mexErrMsgTxt("The second and third inputs should be scalars!");
	}
	if (!mxIsChar(prhs[3]))
	{
		mexErrMsgTxt("The fourth input should be a string!");
	}
	if (nlhs>3) {
		mexErrMsgTxt("Too many output arguments.");
	}

	if (nlhs == 2) deconv = false;
	if (nlhs == 3) deconv = true;

	pointCloud = (float *)mxGetData(prhs[0]);
	size = (int)mxGetM(prhs[0]);
	ndim = (int)mxGetN(prhs[0]);
	maxLevels = (int)mxGetScalar(prhs[1]);
	minPoints = (int)mxGetScalar(prhs[2]);
	nodePointType = mxArrayToString(prhs[3]);

	// copy the point cloud to vector
	vector<Point3f> points3d(size);
	for (int i = 0; i < size; i++) {
		points3d[i].x = pointCloud[i];
		points3d[i].y = pointCloud[size + i];
		points3d[i].z = pointCloud[2 * size + i];

		if (ndim > 3)
		{
			for (int j = 3; j < ndim; j++)
				points3d[i].feat[j - 3] = pointCloud[j * size + i];
		}
	}

	Octree OT(points3d, maxLevels, minPoints);
	OT.computeNodePoint(nodePointType);
	OT.buildInfoMap(deconv);
	
	// assign sorted points to the first output
	mwSize dims[] = { size, ndim };
	plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
	float *sortedPointCloud = (float*)mxGetData(plhs[0]);
	for (int i = 0; i < size; i++) {
		sortedPointCloud[i] = OT.points[i].x;
		sortedPointCloud[size + i] = OT.points[i].y;
		sortedPointCloud[2 * size + i] = OT.points[i].z;

		if (ndim > 3)
		{
			for (int j = 3; j < ndim; j++)
				sortedPointCloud[j * size + i] = OT.points[i].feat[j - 3];
		}
	}
	
	plhs[1] = mxCreateCellMatrix(maxLevels, 1);

	dims[0] = 4;
	for (int l = 0; l < maxLevels; l++)
	{
		dims[1] = OT.infoMap[l].size();
		mxArray *thisMap = mxCreateNumericArray(2, dims, mxUINT32_CLASS, mxREAL);
		unsigned int *thisMapData = (unsigned int *)mxGetData(thisMap);
		for (int k = 0; k < OT.infoMap[l].size(); k++)
		{
			thisMapData[4 * k] = OT.infoMap[l][k].childID;
			thisMapData[4 * k + 1] = OT.infoMap[l][k].filterID;
			thisMapData[4 * k + 2] = OT.infoMap[l][k].parentID;
			thisMapData[4 * k + 3] = OT.infoMap[l][k].convSize;
		}		
		mxSetCell(plhs[1], l, thisMap);
	}

	if(deconv)
	{
		plhs[2] = mxCreateCellMatrix(maxLevels, 1);

		dims[0] = 4;
		for (int l = 0; l < maxLevels; l++)
		{
			dims[1] = OT.infoMap[l].size();
			mxArray *thisMap = mxCreateNumericArray(2, dims, mxUINT32_CLASS, mxREAL);
			unsigned int *thisMapData = (unsigned int *)mxGetData(thisMap);
			for (int k = 0; k < OT.infoMap[l].size(); k++)
			{
				thisMapData[4 * k] = OT.infoMap[l][k].parentID;
				thisMapData[4 * k + 1] = OT.deconvFilter[l][k];
				thisMapData[4 * k + 2] = OT.infoMap[l][k].childID;
				thisMapData[4 * k + 3] = OT.infoMap[l][k].convSize;
			}
			mxSetCell(plhs[2], maxLevels - l - 1, thisMap);
		}
	}
}


