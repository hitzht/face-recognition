#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "face_recognize_api.h"

using namespace cv;
using namespace std;

extern void test(char*, char*);
int main(int argc, char* argv[]) {
	loadModel("../../model/mtcnn_model", "../../model/feature_model/128/model-0000.params", "../../model/feature_model/128/model-symbol.json");
	test(argv[1],argv[2]);
	cv::waitKey();
	return 0;
}
