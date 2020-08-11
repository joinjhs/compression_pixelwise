#include "arithmetic_codec.h"
#include "encode.h"
#include "decode.h"
#include "stdio.h"
#include <time.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <windows.h>
#include <time.h>
#include <fstream>

typedef std::vector<std::string> stringvec;

void read_directory(const std::string& name, stringvec& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void printUsage(char *s) {
	printf("Usage: %s e [source file (ppm)] [compressed file (bin)]\n", s);
	printf("Usage: %s d [compressed file (bin)] [decoded file (ppm)]\n", s);
}
/*
int main(int argc, char *argv[]) {

	char pred[40], context[40];
	strcpy(pred, "0.txt");
	strcpy(context, "0_ctx.txt");

	if (argc == 4 && argv[1][0] == 'e') {

		std::string codename = argv[3];
		std::string code(".bin");
		std::string code_y;
		size_t pos = codename.find(code);
		code_y = codename.replace(pos, codename.length(), "_y.bin");

		char *chr_y = _strdup(code_y.c_str());

		float bpp = 0;

		bpp = runEncoder_pixel(argv[2], chr_y, pred, context);
	}
	else if (argc == 4 && argv[1][0] == 'd') {

		std::string codename = argv[2];
		std::string code(".bin");
		std::string code_y, code_u, code_v;
		size_t pos = codename.find(code);
		code_y = codename.replace(pos, codename.length(), "_y.bin");
		code_u = codename.replace(pos, codename.length(), "_u.bin");
		code_v = codename.replace(pos, codename.length(), "_v.bin");

		char *chr_y = _strdup(code_y.c_str());
		char *chr_u = _strdup(code_u.c_str());
		char *chr_v = _strdup(code_v.c_str());

		runDecoder(chr_y, chr_u, chr_v, argv[3], weights_y, weights_u, weights_v);
	}
	else {
		printUsage(argv[0]);
	}
}
*/

int main() {
	bool encode = true;
	bool decode = true;

	std::string pred_new, context_new, plain_new, out_new, size_new, codename, code(".bin"), code_y;
	char pred[40], context[40], plain[40], out[40], size[40];
	int height, width;
	size_t pos;
	clock_t start, end;
	char* chr_y;

	const int img_num = 18;
	int partition = 4;
	const int row = 9;

	float result[img_num][row];

	printf("img_no bpp_2 encoding_time_2 decoding_time_2 bpp_3 encoding_time_3 decoding_time_3 bpp_4 encoding_time_4 decoding_time_4\n");

	for (int i = 0; i < img_num; i++) {
		printf("%d ", i);
		for (int j = 2; j <= partition; j++) {
			pred_new = "data/" + std::to_string(i) + "_" + std::to_string(j) + "_pred.txt";
			context_new = "data/" + std::to_string(i) + "_" + std::to_string(j) + "_ctx.txt";
			plain_new = "data/" + std::to_string(i) + "_" + std::to_string(j) + ".txt";
			out_new = "data/out_" + std::to_string(i) + "_" + std::to_string(j) + ".txt";
			size_new = "data/" + std::to_string(i) + "_size.txt";
			strcpy(pred, pred_new.c_str());
			strcpy(context, context_new.c_str());
			strcpy(plain, plain_new.c_str());
			strcpy(out, out_new.c_str());
			strcpy(size, size_new.c_str());

			std::ifstream file(size);
			file >> height;
			file >> width;
			codename = "data/compressed.bin";
			pos = codename.find(code);
			code_y = codename.replace(pos, codename.length(), "_"+std::to_string(i)+"_"+std::to_string(j)+".bin");

			chr_y = _strdup(code_y.c_str());

			float bpp = 0;
			if (encode == true) {

				bpp = runEncoder_pixel(plain, chr_y, pred, context, height, width);

			}
			if (decode == true) {
				start = clock();
				runDecoder_pixel(out, chr_y, pred, context);
				end = clock();
				//printf("Total Decoding time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
			}
		}
		printf("\n");
		
		
	}

	
	
	printf("finished");
	system("pause");
}