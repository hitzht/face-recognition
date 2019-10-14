#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "face_recognize_api.h"

using namespace cv;
using namespace std;

extern void test(char*, char*);
int main(int argc, char* argv[]) {
	//int status = loadModel("../../model/mtcnn_model", "../../model/feature_model/512/model-0000.params", "../../model/feature_model/512/model-symbol.json");
	//std::cout << status;
	char* feature_model_params = "model/feature_model/128/model-0000.params";
	char* feature_model_json = "model/feature_model/128/model-symbol.json";

	if (std::string(argv[3]) == "512") {
		feature_model_params = "model/feature_model/512/model-0000.params";
		feature_model_json = "model/feature_model/512/model-symbol.json";
		std::cout << "loading 512 model...." << std::endl;
	}
	
	int status = loadModel("model/mtcnn_model", feature_model_params, feature_model_json);

	test(argv[1],argv[2]);


	cv::waitKey();
	return 0;
}
