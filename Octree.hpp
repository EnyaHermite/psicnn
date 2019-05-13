#include <vector>
#include <string>

#ifndef M_PI 
#define M_PI           3.14159265358979323846F  /* pi */
#endif

#define M_EPS          1e-3             /* eps */
#define MAX_LEAFS 8

using namespace std;


struct Point3f
{
	float x, y, z;
	float feat[20]; // assume each point to have raw features of length no more than 20
};

struct mapID
{
	unsigned int childID, filterID, parentID, convSize;
};

class Octree {
public:
	struct Node
	{
		float x_min, x_max, y_min, y_max, z_min, z_max; //node boundaries
		bool isLeaf; // if the node is a leaf node
		int begin, end; // (end - begin) equals the number of points in this node
		int level;
		Point3f point;
		vector<int> childIDs;
	};
	vector<Node> nodes;
	vector<Point3f> points;
	vector<vector<mapID>> infoMap;
	vector<vector<unsigned int>> deconvFilter;

	int minPoints = 10;
	int maxLevels = 0;
	float maxRadius = 0.0f;

public:
	// Member functions declaration
	Octree(); //empty constructor
	Octree(vector<Point3f>& points3d, int maxLevels, int minPoints);
	void buildTree(vector<Point3f>& points3d, int maxLevels, int minPoints);
	void buildNext(size_t nodeInd);
	void computeNodePoint(string pointType);
	void buildInfoMap(bool deConv);
	~Octree(); //destructor
};