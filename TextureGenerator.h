#ifndef TEXTURE_GENERATOR_H
#define TEXTURE_GENERATOR_H

#include<vector>

class TextureGenerator {
public:
	int width;
	int height;
	int channel;
	std::vector<unsigned char> bytes;

	TextureGenerator(const int width, const int height, const int channel);

	void set_bytes_from_array(std::vector < std::vector<int>> arr);
};


#endif