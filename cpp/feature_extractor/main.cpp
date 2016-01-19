#include <iostream>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;

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
    for (auto it : labels)
    {
        //        cout << it.first << " ";
        //        for (int i = 0; i < 9; i++)
        //            cout << it.second[i] << ' ';
        //        cout << endl;
    }
    return labels;
}

int main()
{
    auto labels = read_labels();
    vector<string> modes = {"train", "test"};
    for (auto mode : modes)
    {
        ifstream in_list(root_path + mode + "_list");
        ifstream in_feat(root_path + mode + "_feat");

        unordered_map<string, vector<string>> photo_to_biz = read_to_biz(root_path + mode + "_photo_to_biz.csv");
        cout << "photos num: " << photo_to_biz.size() << endl;
        cout << "photo per biz: " << photo_per_biz.size() << endl;

        string filename;
        vector<float> feat(2048);
        int cnt = 0;
        unordered_map<string, float[2048]> biz_feat;
        while(getline(in_list, filename))
        {
            in_feat.read(reinterpret_cast<char*>(feat.data()), sizeof(float) * 2048);

            string photo_id = filename.substr(filename.find_last_of("/") + 1);
            photo_id = photo_id.substr(0, photo_id.size() - 4);

            for (size_t i = 0; i < photo_to_biz[photo_id].size(); i++)
            {
                auto &f = biz_feat[photo_to_biz[photo_id][i]];
                for (int i = 0; i < 2048; i++)
                    f[i] += feat[i];
            }
        }

        ofstream out(root_path + mode + "_biz_feat");
        vector<ofstream> out_y(9);
        for (int i = 0; i < 9; i++)
            out_y[i].open(root_path + mode + "_y_" + to_string(i));

        ofstream out_biz(root_path + mode + "_biz");
        for (auto &it : biz_feat)
        {
            out_biz << it.first << endl;
            float image_cnt = photo_per_biz[it.first];
            for (int i = 0; i < 2048; i++)
                it.second[i] /= image_cnt;

            out.write(reinterpret_cast<char*>(&(it.second)), 2048 * 4);

            for (int i = 0; i < 9; i++)
            {
                int y = labels[it.first][i];
                out_y[i].write(reinterpret_cast<char*>(&y), sizeof(y));
            }
            cnt++;
        }
        cout << mode << " complete!" << endl;
        cout << cnt << endl;
    }

    return 0;
}
