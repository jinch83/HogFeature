#include "hog.hpp"

int main(){
  Mat img = imread("~/Desktop/data/a/a001.png");

  HogFeature hog;
  hog.getFeature(img);
  hog.Normalize(1);

  hog.saveFeature("a001_feature.csv");

  return 0;
}

