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
#include "../vl/fisher.h"
#include "../vl/gmm.h"
}

vl_size dimension = 128;
vl_size numCenters = 16;
VlGMM *gmm;

void learn_gmm()
{
	const vl_size numData = 400000;

	vector<float> data(numData * dimension);
	vector<float> buf(dimension);

	ifstream in("/home/dima/yelp/train_feat");
	int q = 0;
	for (q = 0; q < 200000; q++)
	{
		in.read(reinterpret_cast<char*>(buf.data()), dimension * sizeof(float));
		for (size_t i = 0; i < dimension; i++)
			data[q * dimension + i] = buf[i];
	}
	ifstream in2("/home/dima/yelp/test_feat");
	for (q = 200000; q < 400000; q++)
	{
		in2.read(reinterpret_cast<char*>(buf.data()), dimension * sizeof(float));
		for (size_t i = 0; i < dimension; i++)
			data[q * dimension + i] = buf[i];
	}

	gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numCenters);
	vl_gmm_set_initialization(gmm, VlGMMKMeans);
	vl_gmm_cluster (gmm, data.data(), numData);
}

Mat compute_vlad(const Mat &desc)
{
	vector<float> enc(2 * dimension * numCenters);

	vl_fisher_encode(enc.data(), VL_TYPE_FLOAT,
		 vl_gmm_get_means(gmm), dimension, numCenters,
		 vl_gmm_get_covariances(gmm),
		 vl_gmm_get_priors(gmm),
		 reinterpret_cast<float*>(desc.data), desc.rows,
		 VL_FISHER_FLAG_IMPROVED
		 );

	Mat tar(1, 2 * numCenters * dimension, CV_32FC1);
	memcpy(tar.data, enc.data(),  2 * numCenters * dimension * 4);

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
	learn_gmm();

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
