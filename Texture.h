#ifndef TEXTURE_CLASS_H
#define TEXTURE_CLASS_H

#include<glad/glad.h>
#include<vector>

#include"stb_image.h"
#include"shader.h"

class Texture
{
public:
	GLuint ID;
	const char* type;
	GLuint unit;
	int width;
	int height;
	int channel;

	Texture(const char* image, const char* texType, GLuint slot);
	Texture(std::vector<unsigned char> bytes, int widthImg, int heightImg, int numColCh, const char* texType, GLuint slot);


	void update_texture_data(std::vector<unsigned char> data);
	// Assigns a texture unit to a texture
	void texUnit(Shader& shader, const char* uniform, GLuint unit);
	// Binds a texture
	void Bind();
	// Unbinds a texture
	void Unbind();
	// Deletes a texture
	void Delete();
};
#endif