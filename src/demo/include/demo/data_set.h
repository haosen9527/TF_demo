#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

class DataSet {
public:
    DataSet();
    ~DataSet();
    DataSet(string path_name,string image_file,string label_file);
    vector<float>& images() { return images_vec; }
    vector<float>& labels() { return labels_vec; }
    void read_Mnist_Images(string filename, vector<float>&images);
    void read_Mnist_Label(string filename, vector<float>&labels);
    int ReverseInt(int i);

private:
    vector<float> images_vec;
    vector<float> labels_vec;
};
DataSet::DataSet(string path_name, string image_file, string label_file){
  read_Mnist_Images(path_name + image_file, images_vec);
  read_Mnist_Label(path_name + label_file, labels_vec);
  cout<<"End Read Data ..."<<endl;
}
DataSet::~DataSet(){}

/* 读取方法参考
 * https://blog.csdn.net/x_iya/article/details/53052963 */
void DataSet::read_Mnist_Images(string filename, vector<float> &images){
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;

        for (int i = 0; i < number_of_images; i++)
        {
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    images.push_back(image / 255.0);
                }
            }
        }
    }
    else {
      cout<<"file open failed !"<<endl;
    }
}

void DataSet::read_Mnist_Label(string filename, vector<float>&labels){
  ifstream file(filename, ios::binary);
  if (file.is_open())
  {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    cout << "magic number = " << magic_number << endl;
    cout << "number of images = " << number_of_images << endl;

    for (int i = 0; i < number_of_images; i++)
    {
      unsigned char label = 0;
      file.read((char*)&label, sizeof(label));
      float Onehot[10] = {0};
      Onehot[label] = 1; //one-hot编码(int)
      labels.insert( labels.end(), Onehot, Onehot+10);
    }

  }
  else {
    cout<<"file open failed !"<<endl;
  }
}

int DataSet::ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}














