/*
author:jiaopan
date:2019-09-26
email:jiaopaner@163.com
*/

#include "face_recognize_api.h"
#include <opencv2\opencv.hpp>
#include "recognizer.hpp"
#include "cJSON.h"
#include "utils.hpp"
#include <time.h>

using namespace cv;

Recognizer recognizer;
char* extractFaceFeatureByImage(Mat image) {
	cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
	char *resultJson;
	try {
		std::vector<cv::Mat>  aligned_faces = recognizer.createAlignFace(image);
		if (aligned_faces.size() == 0) {
			cJSON_AddNumberToObject(result, "status", -1);
			cJSON_AddStringToObject(result, "msg", "there is no face");
			cJSON_AddItemToObject(result, "embeddings", embeddings);
			resultJson = cJSON_PrintUnformatted(result);
			return resultJson;
		}
		for (int i = 0; i < aligned_faces.size(); i++) {
			cv::Mat features = recognizer.extractFeature(aligned_faces[i]);
			std::vector<double> vector = (std::vector<double>)features;
			
			std::stringstream ss;
			ss << std::setprecision(16);
			std::copy(vector.begin(), vector.end(), std::ostream_iterator<double>(ss, ","));
			std::string values = ss.str();
			values.pop_back();

			cJSON  *embedding;
			cJSON_AddItemToArray(embeddings, embedding = cJSON_CreateObject());
			cJSON_AddStringToObject(embedding, "embedding", values.c_str());
		}
		cJSON_AddNumberToObject(result, "status", 1);
		cJSON_AddStringToObject(result, "msg", "register success");
		cJSON_AddItemToObject(result, "embeddings", embeddings);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	catch (const std::exception&) {
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "register failed");
		cJSON_AddItemToObject(result, "embeddings", embeddings);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
}

char* extractFaceFeature(Mat &face) { 
	resize(face, face, cv::Size(112, 112));
	cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
	char *resultJson;
	try {
		if (face.empty()) {
			cJSON_AddNumberToObject(result, "status", -1);
			cJSON_AddStringToObject(result, "msg", "register failed,there is no face");
			cJSON_AddItemToObject(result, "embeddings", embeddings);
			resultJson = cJSON_PrintUnformatted(result);
			return resultJson;
		}

		cv::Mat features = recognizer.extractFeature(face);
		std::vector<double> vector = (std::vector<double>)features;

		std::stringstream ss;
		ss << std::setprecision(16);
		std::copy(vector.begin(), vector.end(), std::ostream_iterator<double>(ss, ","));
		std::string values = ss.str();
		values.pop_back();
		
		cJSON  *embedding;
		cJSON_AddItemToArray(embeddings, embedding = cJSON_CreateObject());
		cJSON_AddStringToObject(embedding, "embedding", values.c_str());

		cJSON_AddNumberToObject(result, "status", 1);
		cJSON_AddStringToObject(result, "msg", "register success");
		cJSON_AddItemToObject(result, "embeddings", embeddings);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	catch (const std::exception&) {
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "register failed");
		cJSON_AddItemToObject(result, "embeddings", embeddings);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
}
char* computeDistanceByMat(Mat& base, Mat& target,int detected) {
	cJSON  *result = cJSON_CreateObject();
	char *resultJson;
	double distance;
	try {
		/* // todo face aligned
		if (detected == 1) {
			Mat baseAlign, targetAlign;
			resize(base, baseAlign, cv::Size(112, 112));
			resize(target, targetAlign, cv::Size(112, 112));
			Mat base_emb = recognizer.extractFeature(baseAlign);
			Mat target_emb = recognizer.extractFeature(targetAlign);
			distance = recognizer.distance(base_emb, target_emb);
		}
		else {*/
			std::vector<cv::Mat> base_vector = recognizer.createAlignFace(base);
			std::vector<cv::Mat> target_vector = recognizer.createAlignFace(target);

			if ((base_vector.empty() || target_vector.empty()) || (base_vector.size() > 1 || target_vector.size() > 1)) {
				cJSON_AddNumberToObject(result, "status", -1);
				cJSON_AddStringToObject(result, "msg", "compute failed,one of images has no face or has more than one face");
				cJSON_AddNumberToObject(result, "distance", -1);
				cJSON_AddNumberToObject(result, "sim", 0);
				resultJson = cJSON_PrintUnformatted(result);
				return resultJson;
			}
			Mat base_emb = recognizer.extractFeature(base_vector[0]);
			Mat target_emb = recognizer.extractFeature(target_vector[0]);
			distance = recognizer.distance(base_emb, target_emb);
			
			cv::transpose(target_emb, target_emb);
			double sim = base_emb.dot(target_emb);
			if (sim < 0)
				sim = 0;
			if (sim > 100)
				sim = 100;
			
		//}
		cJSON_AddNumberToObject(result, "status", 1);
		cJSON_AddStringToObject(result, "msg", "compute success");
		cJSON_AddNumberToObject(result, "distance",distance);
		cJSON_AddNumberToObject(result, "sim", sim * 100);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	catch (const std::exception&) {
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "compute failed");
		cJSON_AddNumberToObject(result, "distance", -1);
		cJSON_AddNumberToObject(result, "sim", 0);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
}

Mat convertToMat(std::string str) {
	std::vector<double> v;
	std::stringstream ss(str);
	ss << std::setprecision(16);
	std::string token;
	while (std::getline(ss, token, ',')) {
		v.push_back(std::stod(token));
	}
	Mat output = cv::Mat(v, true).reshape(1, 1);
	return output;
}

/*---------------------------------------api list -----------------------------------------------------------------*/
LIB_API int loadModel(char* mtcnn_model,char* insightface_params,char * insightface_json) {
	return recognizer.loadModel(mtcnn_model, insightface_params, insightface_json);
}

LIB_API char * extractFaceFeatureByFile(char * src, int detected = 0){
	cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
	char *resultJson;
	Mat image;
	try{
		image = imread(src);
	}
	catch (const std::exception&){
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "register failed,can not load file");
		cJSON_AddItemToObject(result, "embeddings", embeddings);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	

//	if (detected == 1) {
//		return extractFaceFeature(image); // todo face aligned
//	}
//	else{
		return extractFaceFeatureByImage(image);
//	}
}

LIB_API char * extractFaceFeatureByByte(unsigned char * src, int width, int height, int channels, int detected = 0){
	int format;
	switch (channels) {
	case 1:
		format = CV_8UC1;
		break;
	case 2:
		format = CV_8UC2;
		break;
	case 3:
		format = CV_8UC3;
		break;
	default:
		format = CV_8UC4;
		break;
	}
	Mat image(height, width, format, src);
	//if (detected == 1) {
	//	return extractFaceFeature(image);// todo face aligned
	//}
	//else {
		return extractFaceFeatureByImage(image);
	//}
}

LIB_API char*  extractFaceFeatureByBase64(char* base64_data, int detected = 0) {
	std::string data(base64_data);
	cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
	char *resultJson;
	Mat image;
	try {
		image = Utils::base64ToMat(data);
	}
	catch (const std::exception&) {
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "register failed,can not convert base64 to Mat");
		cJSON_AddItemToObject(result, "embeddings", embeddings);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	//if (detected == 1) {
	//	return extractFaceFeature(image);// todo face aligned
	//}
	//else {
		return extractFaceFeatureByImage(image);
	//}
}


LIB_API char * computeDistance(char * base_emb, char * target_emb){
	cJSON  *result = cJSON_CreateObject();
	char *resultJson;
	try{
		std::string base(base_emb), target(target_emb);
		Mat baseMat = convertToMat(base), targetMat = convertToMat(target);
		double distance = recognizer.distance(baseMat, targetMat);
		cv::transpose(targetMat, targetMat);
		double sim = baseMat.dot(targetMat);
		if (sim < 0)
			sim = 0;
		if (sim > 100)
			sim = 100;
		cJSON_AddNumberToObject(result, "status", 1);
		cJSON_AddStringToObject(result, "msg", "compute success");
		cJSON_AddNumberToObject(result, "distance", distance);
		cJSON_AddNumberToObject(result, "sim", sim * 100);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	catch (const std::exception&){
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "compute failed");
		cJSON_AddNumberToObject(result, "distance", -1);
		cJSON_AddNumberToObject(result, "sim", 0);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}

}
LIB_API char * computeDistanceByFile(char * base_src, char * target_src, int detected = 0){
	cJSON  *result = cJSON_CreateObject();
	char *resultJson;
	Mat base,target;
	try{
		base = imread(base_src);
		target = imread(target_src);
	}
	catch (const std::exception&){
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "can not load file");
		cJSON_AddNumberToObject(result, "distance", -1);
		cJSON_AddNumberToObject(result, "sim", 0);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	return computeDistanceByMat(base, target, detected);
	
	
}
LIB_API char*  computeDistanceByBase64(char* base_data,char* target_data, int detected = 0) {
	std::string base_str(base_data);
	std::string target_str(target_data);
	cJSON  *result = cJSON_CreateObject();
	char *resultJson;
	Mat base, target;
	try {
		base = Utils::base64ToMat(base_str);
		target = Utils::base64ToMat(target_str);
	}
	catch (const std::exception&) {
		cJSON_AddNumberToObject(result, "status", -1);
		cJSON_AddStringToObject(result, "msg", "can not convert base64");
		cJSON_AddNumberToObject(result, "distance", -1);
		cJSON_AddNumberToObject(result, "sim", 0);
		resultJson = cJSON_PrintUnformatted(result);
		return resultJson;
	}
	return computeDistanceByMat(base, target, detected);
}

/*----------------------------------------------------------------------------------------------*/

void test(char* base_src, char* target_src) {
	//Mat base = imread(base_src);
	//Mat target = imread(target_src);
	//Mat base_emb = recognizer.extractFeature(recognizer.createAlignFace(base)[0]);
	////std::cout << base_emb << std::endl;
	//Mat target_emb = recognizer.extractFeature(recognizer.createAlignFace(target)[0]);
	////std::cout << target_emb << std::endl;

	//double dis = recognizer.distance(base_emb, target_emb);
	//std::cout << dis << std::endl;

	//char* result = computeDistance("0.029020833, -0.0068783676, -0.020256473, 0.08922711, 0.1520647, -0.063353509, -0.011182057, 0.23773466, -0.1844321, -0.0074027572, -0.14794661, 0.0051653897, 0.052046064, 0.056903902, 0.079200841, -0.0086160116, -0.10275403, 0.059177171, -0.075488754, -0.049072653, 0.027570521, 0.18733168, -0.010161424, -0.076425344, 0.02510317, -0.086876422, -0.099469766, 0.16163287, -0.084967688, -0.021871576, 0.14369343, 0.097055502, 0.0065840217, -0.018097101, -0.029480744, -0.066096894, 0.016942978, 0.15646647, 0.052293401, 0.12903209, -0.10855469, 0.044652212, 0.00074599194, -0.16131599, -0.072199926, -0.093255699, -0.068642981, 0.12120935, 0.006100629, -0.038449984, 0.073843718, -0.032517787, 0.031847525, -0.0082237236, -0.033020604, -0.026024288, 0.024662545, 0.097049288, -0.013186242, 0.0087926928, 0.085941195, -0.073864095, -0.034101862, -0.062069096, 0.059358981, 0.04966893, -0.036914833, 0.047939252, 0.054796625, -0.018790253, 0.060238205, 0.0076167355, -0.015216754, -0.061193034, -0.016889416, -0.03072207, -0.16774717, 0.068628848, -0.20049851, 0.020299155, 0.1187629, -0.0033529375, -0.030330595, -0.095323272, -0.0049259844, -0.076598287, -0.17594662, -0.073459841, -0.18311255, -0.031051619, 0.03720101, -0.23655726, -0.055039775, 0.012025496, -0.010668803, -0.054880727, -0.0031114377, 0.094407134, -0.03481745, 0.061542794, 0.13756463, 0.14968817, 0.0043450315, 0.042938448, -0.0020956821, 0.10656384, 0.012789681, -0.048712991, 0.0069972454, -0.027091177, -0.063452356, -0.052107193, 0.13972144, 0.0080326824, -0.21505292, -0.043706249, -0.16731535, -0.083271667, -0.055855714, 0.051882118, -0.040529616, 0.091872461, -0.19573945, 0.02434974, 0.037454743, -0.030554138, 0.01387815, 0.18017061","0.09884572, -0.068491496, -0.078664772, 0.032762606, 0.20072919, 0.024014864, 0.047349561, 0.20139465, -0.24945836, 0.016614795, -0.07572569, 0.11808685, 0.17146656, 0.089800715, -0.025433548, 0.040357802, -0.075069338, 0.044914488, -0.093217447, -0.005331621, 0.022735745, -0.058652312, 0.026685696, -0.022456085, 0.02619436, -0.010089835, -0.14127219, 0.14149182, 0.096490413, -0.01297182, 0.20552859, 0.1833231, 0.063791238, -0.054114789, -0.056761984, -0.088626556, -0.057970654, 0.11274455, -0.089882031, 0.036606628, -0.11511264, 0.09308967, 0.059945799, -0.10937923, 0.11097006, -0.019142125, -0.09145803, 0.1311432, 0.1252171, -0.038730226, -0.080530547, 0.081010491, 0.096036114, -0.096100479, 0.049759038, 0.031370837, -0.049491502, 0.13946846, -0.11055151, -0.0038676588, -0.056756437, -0.0038580534, -0.12307825, -0.10289554, -0.0033650277, 0.083691142, 0.028183304, 0.061029408, -0.0049481452, -0.043966934, 0.014331327, 0.03241935, -0.040123675, -0.14365859, -0.014310184, -0.074716471, -0.11835559, 0.082309127, -0.17734039, -0.045208976, 0.10009804, -0.041692022, -0.0055328049, -0.033204131, -0.026514528, -0.10978604, -0.095468342, -0.12418511, -0.15704995, 0.028290421, 0.099584438, -0.19754614, -0.070700161, 0.018390089, -0.062242635, 0.02848771, 0.019649331, 0.043790292, 0.060032263, -0.033581559, 0.10508171, 0.14813991, -0.042962868, -0.026277989, 0.026779169, 0.10178101, 0.037330393, -0.084159054, 0.031589653, -0.00096823327, -0.10374437, -0.052678231, 0.06412217, -0.034865569, -0.040277272, -0.06836205, 0.00085853663, -0.058881979, -0.019776454, 0.036946878, -0.045691911, 0.088631786, -0.11997437, 0.14228344, 0.12539271, -0.0025624367, 0.048964616, 0.05127272");
	//std::cout << result << std::endl;
	
	//char* result = computeDistanceByFile(base_src,target_src,1);
	//std::cout << result << std::endl;

	/*char* features = extractFaceFeatureByFile(base_src);
	std::cout << "f:" << features << std::endl;
	char* temp = extractFaceFeatureByFile(base_src,1);
	std::cout << "tmp:" << temp << std::endl;*/
	//char* features = extractFaceFeatureByFile(base_src);
	//std::cout << "file:" <<features << std::endl;
	/*
	Mat base = imread(base_src);
	std::string base64_data = Mat2Base64(base,"jpg");
	std::cout << "base64:" << base64_data << std::endl;
	Mat image = Base2Mat(base64_data);
	imshow("img",image);
	*/

	/*
	Mat image = imread(base_src);
	char* features = extractFaceFeatureByFile(base_src);
	std::cout << "file:" <<features << std::endl;

	unsigned char* bytes;
	Utils::matToBytes(image, bytes);
	char* result = extractFaceFeatureByByte(bytes, image.cols, image.rows, 3);
	std::cout << "bytes:" << result << std::endl;

	std::fstream fs("test1.txt"); // 创建个文件流对象,并打开"file.txt"
	std::stringstream ss; // 创建字符串流对象
	ss << fs.rdbuf(); // 把文件流中的字符输入到字符串流中
	std::string str = ss.str(); // 获取流中的字符串
	fs.close();
	//char* Base64_features = extractFaceFeatureByBase64(str.c_str());
	//std::cout << "Base64_features:" << Base64_features << std::endl;
	*/
	
	clock_t start, ends;
	start = clock();
	//
	ends = clock();
	std::cout << "result time:" << ends - start << "ms" << std::endl;
	
}
