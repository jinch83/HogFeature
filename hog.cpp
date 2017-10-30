#include "hog.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

HogFeature::HogFeature() {
  _cell_size = 5;
  _block_size = 3;
  isUse = false;
}

HogFeature::HogFeature(int cell_size, int scale){
  _cell_size = cell_size;
  _block_size = scale;
  isUse = false;
}

HogFeature::~HogFeature() {
  if(isUse){
    delete[] cell_feature;
    delete[] hog_feature;
  }
}

void HogFeature::getFeature(Mat img){
  isUse = true;
  Mat gray;
  cvtColor(img, gray, CV_RGB2GRAY);
  _cell_stride = 9; // degree feature
  _block_stride = _cell_stride * _block_size * _block_size;
  //n-dimension
  _cell_dimension = gray.cols*gray.rows*_cell_stride/(_cell_size*_cell_size);
  _hog_dimension = (gray.cols/_cell_size-_block_size+1)*(gray.rows/_cell_size - _block_size+1)*_block_stride;
  cell_feature = new double[_cell_dimension];
  hog_feature = new double[_hog_dimension];
  // initlization
  for(int k=0; k < _cell_dimension; k++){
    cell_feature[k] = 0;
  }
  for(int k=0; k < _hog_dimension; k++){
    hog_feature[k] = 0;
  }
  //end initlization

  // get the pixel info
  int cell_count = 0;
  for(int j=0; j < gray.cols; j=j+_cell_size){//loop start
    for(int i=0; i < gray.rows; i=i+_cell_size){
      // per cells's caculate the histgram about feature to pixel
      for(int u=i; u < min(i+_cell_size, gray.rows); u++){
        double x_vector = 0;
        double y_vector = 0;
        for(int v=j; v < min(j+_cell_size, gray.cols); v++){
          x_vector = (u-1 < 0) ? 0 : (double)gray.at<uchar>(v, u-1);
          y_vector = (v-1 < 0) ? 0 : (double)gray.at<uchar>(v-1, u);

          x_vector = (u + 1 > gray.rows - 1) ? x_vector : (double)gray.at<uchar>(v, u+1) - x_vector;
          y_vector = (v + 1 > gray.cols - 1) ? y_vector : (double)gray.at<uchar>(v+1, u) - y_vector;
        }
        // caculate the feature
        double magnitude = gradient(x_vector, y_vector);
        double degree = tandegree(x_vector, y_vector);
        int num = whatsNum(degree);
        cell_feature[_cell_stride*cell_count + num] += magnitude;
      }
      cell_count++;
      // finish per cell
    }
  }// loop finish
  // create Hog feature
  int _bias_x = 0;
  int _bias_y = 0;
  for(int t=0; t < _hog_dimension/_block_stride; t++){
    for(int u=0; u < _block_size; u++){
      for(int v=0; v < _block_size; v++){
        for(int step=0; step < _cell_stride; step++){
          hog_feature[t*_block_size + 9*(3*u+v) + step] = cell_feature[(6*(u+_bias_y)+v+_bias_x)*_cell_stride + step];
        }
      }
    }
    // y-x tuning
    _bias_x++;
    if(_bias_x > gray.cols/_cell_size-_block_size){
      _bias_x=0;
      _bias_y++;
    }
  }
}

void HogFeature::Normalize(int repeat){
  int count = 0;
  while(count < repeat){
    for(int i=0; i < _hog_dimension; i=i+_block_stride){
      // total checker
      double sum = 0;
      for(int j=0; j < _block_stride; j=j+_cell_stride){
        for(int k=0; k < _cell_stride; k++){
          sum = sum + pow(hog_feature[i+j+k], 2.0);
        }
      }
      sum = sum+1;
      // normalization
      for(int j=0; j < _block_stride; j=j+_cell_stride){
        for(int k=0; k < _cell_stride; k++){
          hog_feature[i+j+k] = hog_feature[i+j+k]/sqrt(sum);
        }
      }
    }
    count++;
  }
}

void HogFeature::saveFeature(string filename){
  ofstream ofs(filename);
  for(int i=0; i < _hog_dimension; i++){
    ofs << i << "," << hog_feature[i] << endl;
  }
}

double HogFeature::gradient(double x, double y) {
  double res = pow(x, 2.0) + pow(y, 2.0);
  res = sqrt(res);

  return res;
}

double HogFeature::tandegree(double x, double y){
  double deg = atan2(y, x) * 180 / 3.14;

  return deg;
}

int HogFeature::whatsNum(double degree) {
  if(degree < 0) {
    degree += 360.0;
  }
  if(degree > 180) {
    degree -= 180;
  }
  degree = degree / 180 + 0.05;
  degree = degree * 10;

  int n = (degree > 8) ? 0 : (int)degree;

  return n;
}
