#ifndef HOG_HPP_
#define HOG_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class HogFeature {
  public:
    HogFeature();
    HogFeature(int, int); // cellxcell size, blockxblock size
    ~HogFeature();
    void getFeature(Mat);
    void Normalize(int); // input the number to repeat the normalization
    void saveFeature(string);
  private:
    int _cell_size;
    int _block_size;
    int _cell_stride;
    int _block_stride;
    int _cell_dimension;
    int _hog_dimension;

    double *cell_feature;
    double *hog_feature;

    double gradient(double, double);
    double tandegree(double, double);
    int whatsNum(double); // input degree!
    bool isUse;
};

#endif
