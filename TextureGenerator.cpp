#include"TextureGenerator.h"
#include<iostream>

TextureGenerator::TextureGenerator(const int width, const int height, const int channel) : width(width), height(height), channel(channel)
{
	bytes.resize(width * height * channel);

	//generate a buffer array
	std::vector<std::vector<int>> arr;
	for (int i = 0; i < height; i++) {
		arr.push_back(std::vector<int>());
		for (int j = 0; j < width; j++) {
			if (i < 5) {
				arr[i].push_back(255);
			}
			else {
				arr[i].push_back(0);
			}
		}
	}

	TextureGenerator::set_bytes_from_array(arr);
}

void TextureGenerator::set_bytes_from_array(std::vector < std::vector<int>> arr) {
	//Sets a 2D array into the bytes
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes[i * width + j] = arr[i][j];
		}
	}
}