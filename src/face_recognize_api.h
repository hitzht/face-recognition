/*
author:jiaopan
date:2019-09-26
email:jiaopaner@163.com
*/
#pragma once
#define LIB_API __declspec(dllexport)

/*
	detected = 0:normal image file that includes faces
	detected = 1:face image that only includes single face
*/
extern "C" {

	LIB_API int loadModel(char* mtcnn_model, char* insightface_params, char * insightface_json);
	
	LIB_API char*  extractFaceFeatureByFile(char* src,int detected);

	LIB_API char*  extractFaceFeatureByByte(unsigned char* src, int width, int height, int channels, int detected);

	/*	
		base64_data:"/9j/4AAQSkZJRgABAQE..."
	*/
	LIB_API char*  extractFaceFeatureByBase64(char* base64_data,int detected);

	/*
		distance < 1:same person or not
		base/target:face features
	*/
	LIB_API char*  computeDistance(char* base,char* target);

	/*
		base/target:image path
	*/
	LIB_API char*  computeDistanceByFile(char* base_src, char* target_src, int detected);

	/*
		base/target:"/9j/4AAQSkZJRgABAQE..."
	*/
	LIB_API char*  computeDistanceByBase64(char* base_data,char* target_data, int detected);
}