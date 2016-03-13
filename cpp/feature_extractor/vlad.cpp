#include <iostream>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

extern "C" {
#include "../vl/mathop.h"
#include "../vl/vlad.h"
#include "../vl/kmeans.h"
}
VlKMeans * kmeans_;
vl_size dimension = 32;
vl_size numCenters = 32;
Mat centers;
Mat compute_vlad(const Mat &desc)
{
	int num_samples_to_encode = desc.rows;

	float *data_to_encode = (float*)vl_malloc(sizeof(float) * dimension * num_samples_to_encode);

	for (int i = 0; i < desc.rows; i++)
		for (int j = 0; j < desc.cols; j++)
			data_to_encode[i * dimension + j] = desc.at<float>(i, j);

	vl_uint32 *indexes = (vl_uint32*)vl_malloc(sizeof(vl_uint32) * num_samples_to_encode);
	float *distances = (float*)vl_malloc(sizeof(float) * num_samples_to_encode);

	vl_kmeans_quantize(kmeans_, indexes, distances, data_to_encode, num_samples_to_encode);

	float *assignments2 = (float*)vl_malloc(sizeof(float) * num_samples_to_encode * numCenters);
	memset(assignments2, 0, sizeof(float) * num_samples_to_encode * numCenters);

	for(int i = 0; i < num_samples_to_encode; i++)
	{
		assignments2[i * numCenters + indexes[i]] = 1.;
	}

	float *enc = (float*)vl_malloc(sizeof(float) * dimension * numCenters);
	vl_vlad_encode (enc, VL_TYPE_FLOAT, reinterpret_cast<float*>(centers.data), dimension, numCenters, data_to_encode, num_samples_to_encode, assignments2, VL_VLAD_FLAG_NORMALIZE_COMPONENTS) ;

	Mat tar(1, numCenters * dimension, CV_32FC1);
	tar.data = (uchar*)enc;

	return tar;
}

string root_path = "/home/dima/yelp/";
unordered_map<string, int> photo_per_biz;

unordered_map<string, vector<string>> read_to_biz(string filename)
{
	unordered_map<string, vector<string>> photo_to_biz;
	ifstream in_to_biz(filename);
	string line;
	getline(in_to_biz, line);
	while(getline(in_to_biz, line))
	{
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of(","));
		photo_to_biz[tokens[0]].push_back(tokens[1]);
		photo_per_biz[tokens[1]]++;
	}
	return photo_to_biz;
}

unordered_map<string, int[9]> read_labels()
{
	ifstream in(root_path + "train.csv");
	string line;
	getline(in, line);
	unordered_map<string, int[9]> labels;
	while(getline(in, line))
	{
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of(","));
		string biz = tokens[0];
		string list = tokens[1];
		boost::split(tokens, list, boost::is_any_of(" "));
		try
		{
			for (size_t i = 0; i < tokens.size(); i++)
				labels[biz][stoi(tokens[i])] = 1;
		}
		catch(...)
		{
			cout << "empty line: " << line << endl;
		}
	}
	return labels;
}

int main()
{
	centers = Mat(numCenters, dimension, CV_32FC1);
	ifstream inq(root_path + "/centers_" + to_string(dimension) + "_" + to_string(numCenters));
	for (size_t i = 0; i < numCenters; i++)
	{
		inq.read(reinterpret_cast<char*>(centers.row(i).data), sizeof(float) * dimension);
	}
//	centers = centers.t();

	VlVectorComparisonType distance = VlDistanceL2 ;
	kmeans_ = vl_kmeans_new (VL_TYPE_FLOAT,distance);
	vl_kmeans_set_centers(kmeans_, reinterpret_cast<float*>(centers.data), dimension, numCenters);

	cout << centers << endl;

	auto labels = read_labels();
	vector<string> modes = {"train", "test"};
	for (auto mode : modes)
	{
		ifstream in_list(root_path + mode + "_list");
		ifstream in_feat(root_path + mode + "_feat");

		photo_per_biz.clear();
		unordered_map<string, vector<string>> photo_to_biz = read_to_biz(root_path + mode + "_photo_to_biz.csv");
		cout << "photos num: " << photo_to_biz.size() << endl;
		cout << "photo per biz: " << photo_per_biz.size() << endl;

		string filename;
		Mat feat(1, dimension, CV_32FC1);
		int cnt = 0;
		unordered_map<string, Mat> business_feat;
		while(getline(in_list, filename))
		{
			in_feat.read(reinterpret_cast<char*>(feat.data), sizeof(float) * feat.total());

			string photo_id = filename.substr(filename.find_last_of("/") + 1);
			photo_id = photo_id.substr(0, photo_id.size() - 4);

			for (size_t i = 0; i < photo_to_biz[photo_id].size(); i++)
			{
				auto &f = business_feat[photo_to_biz[photo_id][i]];
				f.push_back(feat);
			}
		}
		ofstream out(root_path + "/vlad_" + mode);

		ofstream out_biz(root_path + "/vlad_business_" + mode);
		for (auto &it : business_feat)
		{
			out_biz << it.first << endl;
			Mat feat = it.second;
			Mat vlad = compute_vlad(feat);

			out.write(reinterpret_cast<char*>(vlad.data), vlad.total() * sizeof(float));
			cnt++;
		}
		cout << mode << " complete!" << endl;
		cout << cnt << endl;
	}

	return 0;
}
