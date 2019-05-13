#include "Octree.hpp"
#include <math.h>
#include <assert.h>
#include <limits>
#include <cmath>


#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif


/* -------------------------------------------------------------------
* Helpers for the Octree construction
* ---------------------------------------------------------------- */

void fillMinMax(const vector<Point3f>& points, Octree::Node& node)
{
	node.x_max = node.y_max = node.z_max = std::numeric_limits<float>::min();
	node.x_min = node.y_min = node.z_min = std::numeric_limits<float>::max();

	for (size_t i = 0; i < points.size(); ++i)
	{
		const Point3f& point = points[i];

		if (node.x_max < point.x)
			node.x_max = point.x;

		if (node.y_max < point.y)
			node.y_max = point.y;

		if (node.z_max < point.z)
			node.z_max = point.z;

		if (node.x_min > point.x)
			node.x_min = point.x;

		if (node.y_min > point.y)
			node.y_min = point.y;

		if (node.z_min > point.z)
			node.z_min = point.z;
	}
}

size_t findSubboxForPoint(const Point3f& point, const Octree::Node& node)
{
	size_t ind_x = point.x < (node.x_max + node.x_min) / 2 ? 0 : 1;
	size_t ind_y = point.y < (node.y_max + node.y_min) / 2 ? 0 : 1;
	size_t ind_z = point.z < (node.z_max + node.z_min) / 2 ? 0 : 1;

	return (ind_x << 2) + (ind_y << 1) + (ind_z << 0);
}

void initChildBox(const Octree::Node& parent, size_t boxIndex, Octree::Node& child)
{
	child.x_min = child.x_max = (parent.x_max + parent.x_min) / 2;
	child.y_min = child.y_max = (parent.y_max + parent.y_min) / 2;
	child.z_min = child.z_max = (parent.z_max + parent.z_min) / 2;

	if ((boxIndex >> 0) & 1) //a&b: bit-wise and, a^b: bit-wise xor
		child.z_max = parent.z_max;
	else
		child.z_min = parent.z_min;

	if ((boxIndex >> 1) & 1)
		child.y_max = parent.y_max;
	else
		child.y_min = parent.y_min;

	if ((boxIndex >> 2) & 1)
		child.x_max = parent.x_max;
	else
		child.x_min = parent.x_min;
}


// -----------------------------------------------------------------------
//                                                     Octree construction
// -----------------------------------------------------------------------

Octree::Octree() {} // empty constructor 

Octree::~Octree()
{
	nodes.clear();
	points.clear();
}  // destructor

Octree::Octree(vector<Point3f>& points3d, int _maxLevels, int _minPoints)
{
	buildTree(points3d, _maxLevels, _minPoints);
}

void Octree::buildTree(vector<Point3f>& points3d, int _maxLevels, int _minPoints)
{
	points = points3d; // initialization
	minPoints = _minPoints;
	maxLevels = _maxLevels;

	nodes.clear();
	nodes.reserve(50000);
	nodes.push_back(Node()); // Node() 
	Node& root = nodes[0];
	fillMinMax(points, root);

	maxRadius = sqrt((root.x_max - root.x_min)*(root.x_max - root.x_min) +
					 (root.y_max - root.y_min)*(root.y_max - root.y_min) +
					 (root.z_max - root.z_min)*(root.z_max - root.z_min))/2;

	root.isLeaf = true;
	root.level = maxLevels;
	root.begin = 0;
	root.end = (int)points.size();
	root.childIDs.reserve(MAX_LEAFS);

	if (maxLevels != 1 && (root.end - root.begin) > _minPoints)
	{
		root.isLeaf = false;
		buildNext(0);
	}
}

void  Octree::buildNext(size_t nodeInd)
{
	size_t size = nodes[nodeInd].end - nodes[nodeInd].begin;

	vector<size_t> boxBorders(MAX_LEAFS + 1, 0); 
	vector<size_t> boxIndices(size); // indices of each point in the point cloud
	vector<Point3f> tempPoints(size); //a sorted copy of the point cloud

	for (int i = nodes[nodeInd].begin, j = 0; i < nodes[nodeInd].end; ++i, ++j)
	{
		const Point3f& p = points[i];

		int subboxInd = findSubboxForPoint(p, nodes[nodeInd]);

		boxBorders[subboxInd + 1]++;//1~8: the leaf of current nodes
		boxIndices[j] = subboxInd;
		tempPoints[j] = p;
	}

	for (size_t i = 1; i < boxBorders.size(); ++i)
		boxBorders[i] += boxBorders[i - 1];

	vector<size_t> writeInds(boxBorders.begin(), boxBorders.end());

	for (size_t i = 0; i < size; ++i)
	{
		size_t boxIndex = boxIndices[i];
		Point3f& curPoint = tempPoints[i];

		size_t copyTo = nodes[nodeInd].begin + writeInds[boxIndex]++;
		points[copyTo] = curPoint;
	} // sort original point from smallest children id(0) to maximum id(7)


	for (size_t i = 0; i < MAX_LEAFS; ++i)
	{
		if (boxBorders[i] == boxBorders[i + 1]) // id=i is null node, null node is 												
			continue;							// not included in the octree nodes

		nodes.push_back(Node()); //initialize a node
		Node& child = nodes.back(); // return the last node in nodes
		initChildBox(nodes[nodeInd], i, child);

		child.isLeaf = true;
		child.level = nodes[nodeInd].level - 1;
		child.begin = nodes[nodeInd].begin + (int)boxBorders[i + 0]; //beginning index of points in this child node 
																	 // in the whole points(sorted with writeInds)
		child.end = nodes[nodeInd].begin + (int)boxBorders[i + 1]; //[begin, end) left-closed, right-open
		child.childIDs.reserve(MAX_LEAFS);

		int childID = (int)(nodes.size() - 1);
		nodes[nodeInd].childIDs.push_back(childID); //children node index in nodes  

		if (child.level != 1 && (child.end - child.begin) > minPoints)
		{
			child.isLeaf = false;
			buildNext(childID);
		}
	}
}

// -----------------------------------------------------------------------
//                          Computing typical location of each Octree node
// -----------------------------------------------------------------------

void Octree::computeNodePoint(string pointType)
{
	if (pointType == "center")
	{
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i].point.x = (nodes[i].x_min + nodes[i].x_max) / 2;
			nodes[i].point.y = (nodes[i].y_min + nodes[i].y_max) / 2;
			nodes[i].point.z = (nodes[i].z_min + nodes[i].z_max) / 2;
		}
	}
	if (pointType == "childAverage")
	{
		//compute points of leaf nodes
		for (int i = 0; i < nodes.size(); i++)
		{
			if (nodes[i].isLeaf)
			{
				nodes[i].point.x = 0.0f;
				nodes[i].point.y = 0.0f;
				nodes[i].point.z = 0.0f;
				size_t size = nodes[i].end - nodes[i].begin;
				for (int j = nodes[i].begin; j < nodes[i].end; j++)
				{
					nodes[i].point.x += points[j].x;
					nodes[i].point.y += points[j].y;
					nodes[i].point.z += points[j].z;
				}
				nodes[i].point.x /= size;
				nodes[i].point.y /= size;
				nodes[i].point.z /= size;
			}
		}

		// compute point of parent nodes from child nodes
		for (int level = 2; level <= nodes[0].level; level++)
		{
			for (int i = 0; i < nodes.size(); i++)
			{
				if (nodes[i].level == level & !nodes[i].isLeaf)
				{
					nodes[i].point.x = 0.0f;
					nodes[i].point.y = 0.0f;
					nodes[i].point.z = 0.0f;
					size_t size = nodes[i].childIDs.size();
					for (int j = 0; j < nodes[i].childIDs.size(); j++)
					{
						int childID = nodes[i].childIDs[j];
						nodes[i].point.x += nodes[childID].point.x;
						nodes[i].point.y += nodes[childID].point.y;
						nodes[i].point.z += nodes[childID].point.z;
					}
					nodes[i].point.x /= size;
					nodes[i].point.y /= size;
					nodes[i].point.z /= size;
				}
			}
		}
	}
	if (pointType == "pointAverage")
	{
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i].point.x = 0.0f;
			nodes[i].point.y = 0.0f;
			nodes[i].point.z = 0.0f;
			size_t size = (nodes[i].end - nodes[i].begin);
			for (int j = nodes[i].begin; j < nodes[i].end; j++)
			{
				nodes[i].point.x += points[j].x;
				nodes[i].point.y += points[j].y;
				nodes[i].point.z += points[j].z;
			}
			nodes[i].point.x /= size;
			nodes[i].point.y /= size;
			nodes[i].point.z /= size;
		}
	}
}



/* -------------------------------------------------------------------
* Helpers for building the Octree map
* ---------------------------------------------------------------- */

// cartesian coordinates to spherical coordinates
void cart2sph(const Point3f& reLoc, float& theta, float& phi, float& radius)
{
	radius = sqrt(reLoc.x*reLoc.x + reLoc.y*reLoc.y + reLoc.z*reLoc.z);
	theta = atan2(reLoc.y, reLoc.x);
	phi = atan2(reLoc.z, sqrt(reLoc.x*reLoc.x + reLoc.y*reLoc.y));
}

// build bin edges
void buildEdges(vector<float>& thetaEdges, vector<float>& phiEdges, vector<float>& radiusEdges)
{
	int numTheta = 2 * 4 + 1; // angular step for theta is pi/4
	int numPhi = 2 + 1; // angular step for phi is pi/2
	int numRadius = 4;
	thetaEdges.resize(numTheta);
	phiEdges.resize(numPhi);
	radiusEdges.resize(numRadius);

	for (int i = 0; i < thetaEdges.size(); i++)
	{
		thetaEdges[i] = i*(M_PI / 4) - M_PI; // [-pi, pi]
	}
	thetaEdges.front() -= M_EPS;
	thetaEdges.back() += M_EPS;

	for (int i = 0; i < phiEdges.size(); i++)
	{
		phiEdges[i] = i*(M_PI / 2) - M_PI / 2; // [-pi/2, pi/2]
	}
	phiEdges.front() -= M_EPS;
	phiEdges.back() += M_EPS;

	for (int i = 0; i < radiusEdges.size()-1; i++)
	{
		radiusEdges[i] = float(i)/float(numRadius - 1); // [0, 1/3, 2/3, 2]
	}
	radiusEdges.back() = 2;
}

//Compute filterID of the child relative to its parent
int getFilterID(const Point3f& center, const Point3f& point, const float& normRadius,
	const vector<float>& thetaEdges, const vector<float>& phiEdges, const vector<float>& radiusEdges)
{
	int ID, m, n, p; // not in any bins

	Point3f reLoc;
	float theta, phi, radius;
	reLoc.x = point.x - center.x;
	reLoc.y = point.y - center.y;
	reLoc.z = point.z - center.z;
	cart2sph(reLoc, theta, phi, radius);
	radius /= normRadius;

	//if in the first bin: self-convolution
	float selfR = M_EPS; 
	if (radius < selfR)
	{
		ID = 0;
		return ID;
	}

	for (m = 0; m < radiusEdges.size() - 1; m++)
	{
		if (radius >= radiusEdges[m] & radius < radiusEdges[m + 1]) { break; }
	}
	
	for (n = 0; n < phiEdges.size() - 1; n++)
	{
		if (phi >= phiEdges[n] & phi < phiEdges[n + 1]) { break; }
	}

	for (p = 0; p < thetaEdges.size() - 1; p++)
	{
		if (theta >= thetaEdges[p] & theta < thetaEdges[p + 1]) { break; }
	}

	ID = m * (thetaEdges.size() - 1) * (phiEdges.size() - 1) +
		 n * (thetaEdges.size() - 1) + p + 1;
	return ID;
}


// -----------------------------------------------------------------------
//                                                 Building the Octree map 
// -----------------------------------------------------------------------

void Octree::buildInfoMap(bool deConv)
{
	vector<float> thetaEdges, phiEdges, radiusEdges;
	buildEdges(thetaEdges, phiEdges, radiusEdges);

	infoMap.resize(maxLevels);
	if (deConv) deconvFilter.resize(maxLevels);

	// compute the infoMap
	unsigned int filterID, convSize;
	vector<unsigned int> nodeID(nodes.size(), 0); // ID of each node in its own level, 0-based consecutive indexing
	vector<unsigned int> levelID(maxLevels + 1, 0); // current index of each level in the octree, one additional for the input

	for (int l = 1; l <= maxLevels; l++)
	{
		infoMap[l - 1].reserve(points.size());
		if (deConv) deconvFilter[l - 1].reserve(points.size());
	}

	vector<Point3f> tempPoints(points.size());
	
	for (int j = 0; j < nodes.size(); j++)
	{
		int l = nodes[j].level;
		float thisRadius = maxRadius / (float)(1 << maxLevels - l);

		if (nodes[j].isLeaf) // case 1: nodes[j] is a leaf node
		{
			convSize = nodes[j].end - nodes[j].begin;
			for (int k = nodes[j].begin; k < nodes[j].end; k++)
			{
				filterID = getFilterID(nodes[j].point, points[k], thisRadius, thetaEdges, phiEdges, radiusEdges);
				infoMap[l - 1].push_back({ levelID[l - 1], filterID, nodeID[j], convSize });

				if (deConv)
				{
					filterID = getFilterID(points[k], nodes[j].point, thisRadius, thetaEdges, phiEdges, radiusEdges);
					deconvFilter[l - 1].push_back(filterID);
				}

				const Point3f& P = points[k];
				tempPoints[levelID[0]] = P;

				for (int pl = l - 1; pl > 0; pl--)
				{
					infoMap[pl - 1].push_back({ levelID[pl - 1], 0, levelID[pl], 1 }); // self-loop convolution																						 
					if (deConv) deconvFilter[pl - 1].push_back(0);
				}

				for (int pl = l; pl > 0; pl--)
				{
					levelID[pl - 1]++;
				}
			}
		}
		else // condition 2: node[j] is a non-leaf node
		{
			convSize = nodes[j].childIDs.size();
			for (int k = 0; k < nodes[j].childIDs.size(); k++)
			{
				int child = nodes[j].childIDs[k];
				nodeID[child] = levelID[l - 1];
				filterID = getFilterID(nodes[j].point, nodes[child].point, thisRadius, thetaEdges, phiEdges, radiusEdges);
				infoMap[l - 1].push_back({ nodeID[child], filterID, nodeID[j], convSize });

				if (deConv)
				{
					filterID = getFilterID(nodes[child].point, nodes[j].point, thisRadius, thetaEdges, phiEdges, radiusEdges);
					deconvFilter[l - 1].push_back(filterID);
				}

				levelID[l - 1]++;
			}
		}
	}

	points = tempPoints;
	tempPoints.clear();
}
