/*
author:jiaopan
date:2019-09-26
email:jiaopaner@163.com
*/
#pragma once
#define LIB_API __declspec(dllexport)

extern "C" {

	LIB_API int loadModel(char* mtcnn_model, char* insightface_params, char * insightface_json);
	
	/*
		 detected = 0:normal image file that includes faces
		 detected = 1:face image that only includes single face
	*/
	LIB_API char*  extractFaceFeatureByFile(char* src,int detected);

	/*
		 detected = 0:normal image file that includes faces
		 detected = 1:face image that only includes single face
	*/
	LIB_API char*  extractFaceFeatureByByte(unsigned char* src, int width, int height, int channels, int detected);

	/*
		distance < 1:same person or not
		base/target:face features
	*/
	LIB_API char*  computeDistance(char* base,char* target);

	/*
		detected = 0:normal image file that includes faces
		detected = 1:face image that only includes single face
		base/target:image path
	*/
	LIB_API char*  computeDistanceByFile(char* base_src, char* target_src, int detected);
}