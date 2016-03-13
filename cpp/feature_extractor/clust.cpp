#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

extern "C" {
#include "../vl/mathop.h"
#include "../vl/vlad.h"
#include "../vl/kmeans.h"
#include "../vl/random.h"
}

using namespace std;
using namespace cv;

using ull = unsigned long long;

int main()
{
	const vl_size numData = 230000;
	const vl_size dimension = 32;
	vl_size numCenters = 32;
	vl_size maxiter = 1;
	vl_size maxComp = 50;
	vl_size maxrep = 1;
	vl_size ntrees = 10;

	ifstream in("/home/dima/yelp/train_feat");
	float *data = new float[numData * dimension];
	float buf[dimension];

	for (size_t q = 0; q < numData; q++)
	{
		in.read(reinterpret_cast<char*>(&buf), dimension * sizeof(float));
		for (size_t i = 0; i < dimension; i++)
			data[q * dimension + i] = buf[i];
	}
	cout << "reading completed" << endl;
	auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

//		VlKMeansAlgorithm algorithm = VlKMeansANN;
		VlKMeansAlgorithm algorithm = VlKMeansLloyd;
//    VlKMeansAlgorithm algorithm = VlKMeansElkan;

	VlVectorComparisonType distance = VlDistanceL2;
	VlKMeans * kmeans_ = vl_kmeans_new (VL_TYPE_FLOAT,distance);

	vl_kmeans_set_verbosity	(kmeans_, 1);
	vl_kmeans_set_max_num_iterations (kmeans_, maxiter) ;
	vl_kmeans_set_max_num_comparisons (kmeans_, maxComp) ;
	vl_kmeans_set_num_repetitions (kmeans_, maxrep) ;
	vl_kmeans_set_num_trees (kmeans_, ntrees);
	vl_kmeans_set_algorithm (kmeans_, algorithm);
	vl_set_num_threads(8);
//	vl_kmeans_set_initialization(kmeans_, VlKMeansRandomSelection);
	vl_kmeans_set_initialization(kmeans_, VlKMeansPlusPlus);

	srand(time(0));

	vl_kmeans_cluster(kmeans_, data, dimension, numData, numCenters);
	cout << "in:: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start << endl;

	float *centers = (float*)vl_kmeans_get_centers(kmeans_);
	ofstream file_centers("/home/dima/yelp/centers_" + to_string(dimension) + "_" + to_string(numCenters));

	for (size_t i = 0; i < numCenters; i++)
		file_centers.write(reinterpret_cast<char*>(centers + i * dimension), sizeof(float) * dimension);

	return 0;
}

