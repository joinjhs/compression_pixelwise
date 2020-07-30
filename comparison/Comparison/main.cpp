#include "stdio.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <windows.h>
#include <fstream>

#define CLASSIC 0
#define KODAK 1
#define MCM 2
#define WSI 3
#define OPENIMAGES 4
#define OPENIMAGES_SMALL 5
#define DIV2K 6
#define DIV2K_SMALL 7

#pragma comment(lib, "Ws2_32.lib")

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

bool replace(std::string& str, const std::string& from, const std::string& to) {
	size_t start_pos = str.find(from);
	if (start_pos == std::string::npos)
		return false;
	str.replace(start_pos, from.length(), to);
	return true;
}


void main(int argc, char *argv[]) {

	bool TIME = false;
	bool RUN_BPG = true;
	bool RUN_PNG = true;
	bool RUN_WEBP = true;
	bool RUN_FLIF = true;

	int data_type = MCM;

	std::string dir_name;
	std::string dir_time = "data/wsi/";

	if (data_type == CLASSIC)
		dir_name = "data/classic/";
	else if(data_type == KODAK)
		dir_name = "data/kodak/";
	else if (data_type == MCM)
		dir_name = "data/mcm/";
	else if (data_type == WSI)
		dir_name = "data/wsi/";
	else if (data_type == OPENIMAGES)
		dir_name = "data/open_images/";
	else if (data_type == OPENIMAGES_SMALL)
		dir_name = "data/open_images_small/";
	else if (data_type == DIV2K)
		dir_name = "data/div2k/";
	else if (data_type == DIV2K_SMALL)
		dir_name = "data/div2k_small/";

	stringvec v, v_time;

	read_directory(dir_name, v);
	v.erase(v.begin(), v.begin() + 2);

	read_directory(dir_time, v_time);
	v_time.erase(v_time.begin(), v_time.begin() + 2);

	int num_files = v.size();
	int num_files_time = v_time.size();
	float avg_bpp = 0;
	double avg_enc_t = 0, avg_dec_t = 0, avg_t = 0;

	std::string infile;
	char infilename[40];
	char command[128];
	struct stat st;
	float cur_bpp;
	double cur_enc_t, cur_dec_t, cur_t;
	clock_t start, end;

	if (RUN_BPG) {

		printf("========== BPG ==========\n");
		// BPP Performance
		for (int i = 0; i < num_files; i++) {

			// Encoding
			infile = dir_name + v.at(i);
			strcpy(infilename, infile.c_str());

			sprintf(command, "bpgenc.exe -lossless %s", infilename);
			system(command);

			// Calculate BPP
			std::ifstream in(infilename);
			unsigned int width, height;

			in.seekg(16);
			in.read((char *)&width, 4);
			in.read((char *)&height, 4);

			width = ntohl(width);
			height = ntohl(height);

			printf("%s  ", infilename);

			stat("out.bpg", &st);

			cur_bpp = 8.0 * st.st_size / (width * height);
			printf("BPP : %f\n", cur_bpp);

			avg_bpp += cur_bpp;
		}

		printf("Average bpp : %f\n", avg_bpp / num_files);

		// Time Computation
		if (TIME) {
			
			for (int i = 0; i < num_files_time; i++) {
				// Encoding
				infile = dir_time + v_time.at(i);
				strcpy(infilename, infile.c_str());

				sprintf(command, "bpgenc.exe -lossless %s", infilename);
				start = clock();
				system(command);
				end = clock();

				cur_enc_t = (end - start) / (double)CLOCKS_PER_SEC;

				// Decoding
				sprintf(command, "bpgdec.exe -o out.png out.bpg");
				start = clock();
				system(command);
				end = clock();

				cur_dec_t = (end - start) / (double)CLOCKS_PER_SEC;

				cur_t = cur_enc_t + cur_dec_t;

				avg_enc_t += cur_enc_t;
				avg_dec_t += cur_dec_t;
				avg_t += cur_t;

				printf("%s  ", infilename);
				printf("Enc / Dec / Total : %f %f %f\n", cur_enc_t, cur_dec_t, cur_t);
			}

			printf("Average time (Enc / Dec / Total) : %f %f %f\n", avg_enc_t / num_files_time, avg_dec_t / num_files_time , avg_t / num_files_time);
		}
	}

	if (RUN_PNG) {

		avg_bpp = 0;
		avg_enc_t = 0;
		avg_dec_t = 0;
		avg_t = 0;

		printf("========== PNG ==========\n");

		// BPP Performance
		for (int i = 0; i < num_files; i++) {
			infile = dir_name + v.at(i);
			strcpy(infilename, infile.c_str());

			// Calculate BPP
			std::ifstream in(infilename);
			unsigned int width, height;

			in.seekg(16);
			in.read((char *)&width, 4);
			in.read((char *)&height, 4);

			width = ntohl(width);
			height = ntohl(height);

			printf("%s  ", infilename);

			stat(infilename, &st);

			cur_bpp = 8.0 * st.st_size / (width * height);
			printf("BPP : %f\n", cur_bpp);

			avg_bpp += cur_bpp;
		}

		printf("Average bpp : %f\n", avg_bpp / num_files);
	}

	if (RUN_WEBP) {

		avg_bpp = 0;
		avg_enc_t = 0;
		avg_dec_t = 0;
		avg_t = 0;

		printf("========== WebP ==========\n");

		// BPP Performance
		for (int i = 0; i < num_files; i++) {

			// Encoding
			infile = dir_name + v.at(i);
			strcpy(infilename, infile.c_str());

			sprintf(command, "cwebp.exe %s -lossless -m 6 -q 100 -o out.webp", infilename);
			system(command);

			// Calculate BPP
			std::ifstream in(infilename);
			unsigned int width, height;

			in.seekg(16);
			in.read((char *)&width, 4);
			in.read((char *)&height, 4);

			width = ntohl(width);
			height = ntohl(height);

			printf("%s  ", infilename);

			stat("out.webp", &st);

			cur_bpp = 8.0 * st.st_size / (width * height);
			printf("BPP : %f\n", cur_bpp);

			avg_bpp += cur_bpp;
		}

		printf("Average bpp : %f\n", avg_bpp / num_files);

		// Time Computation
		if (TIME) {

			for (int i = 0; i < num_files_time; i++) {
				// Encoding
				infile = dir_time + v_time.at(i);
				strcpy(infilename, infile.c_str());

				sprintf(command, "cwebp.exe %s -lossless -m 6 -q 100 -o out.webp", infilename);
				start = clock();
				system(command);
				end = clock();

				cur_enc_t = (end - start) / (double)CLOCKS_PER_SEC;

				// Decoding
				sprintf(command, "dwebp.exe out.webp -o out.png");
				start = clock();
				system(command);
				end = clock();

				cur_dec_t = (end - start) / (double)CLOCKS_PER_SEC;

				cur_t = cur_enc_t + cur_dec_t;

				avg_enc_t += cur_enc_t;
				avg_dec_t += cur_dec_t;
				avg_t += cur_t;

				printf("%s  ", infilename);
				printf("Enc / Dec / Total : %f %f %f\n", cur_enc_t, cur_dec_t, cur_t);
			}

			printf("Average time (Enc / Dec / Total) : %f %f %f\n", avg_enc_t / num_files_time, avg_dec_t / num_files_time, avg_t / num_files_time);
		}
	}

	if (RUN_FLIF) {

		avg_bpp = 0;
		avg_enc_t = 0;
		avg_dec_t = 0;
		avg_t = 0;

		printf("========== FLIF ==========\n");

		// BPP Performance
		for (int i = 0; i < num_files; i++) {

			remove("out.flif");

			// Encoding
			infile = dir_name + v.at(i);
			strcpy(infilename, infile.c_str());

			sprintf(command, "flif.exe -e -Q100 %s out.flif", infilename);
			system(command);

			// Calculate BPP
			std::ifstream in(infilename);
			unsigned int width, height;

			in.seekg(16);
			in.read((char *)&width, 4);
			in.read((char *)&height, 4);

			width = ntohl(width);
			height = ntohl(height);

			printf("%s  ", infilename);

			stat("out.flif", &st);

			cur_bpp = 8.0 * st.st_size / (width * height);
			printf("BPP : %f\n", cur_bpp);

			avg_bpp += cur_bpp;
		}

		printf("Average bpp : %f\n", avg_bpp / num_files);

		// Time Computation
		if (TIME) {

			for (int i = 0; i < num_files_time; i++) {
				remove("out.flif");

				// Encoding
				infile = dir_time + v_time.at(i);
				strcpy(infilename, infile.c_str());

				sprintf(command, "flif.exe -e -Q100 %s out.flif", infilename);
				start = clock();
				system(command);
				end = clock();

				cur_enc_t = (end - start) / (double)CLOCKS_PER_SEC;

				// Decoding
				sprintf(command, "flif.exe -d out.flif out.png");
				start = clock();
				system(command);
				end = clock();

				cur_dec_t = (end - start) / (double)CLOCKS_PER_SEC;

				cur_t = cur_enc_t + cur_dec_t;

				avg_enc_t += cur_enc_t;
				avg_dec_t += cur_dec_t;
				avg_t += cur_t;

				printf("%s  ", infilename);
				printf("Enc / Dec / Total : %f %f %f\n", cur_enc_t, cur_dec_t, cur_t);
			}

			printf("Average time (Enc / Dec / Total) : %f %f %f\n", avg_enc_t / num_files_time, avg_dec_t / num_files_time, avg_t / num_files_time);
		}
	}
	system("pause");
	}