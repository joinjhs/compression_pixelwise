#include "arithmetic_codec.h"
#include "encode.h"
#include "decode.h"
#include "stdio.h"
#include "ppm_io.h"
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
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

typedef std::vector<std::string> stringvec;
namespace np = boost::python::numpy;
namespace nb = boost::python;

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

void test() {
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

	//std::ofstream record("record.txt", std::ios::app);
	//record << "img_no bpp_2 encoding_time_2 decoding_time_2 bpp_3 encoding_time_3 decoding_time_3 bpp_4 encoding_time_4 decoding_time_4"<<std::endl;

	for (int i = 0; i < img_num; i++) {
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
		std::ofstream record("record.txt", std::ios::app);
		record << std::endl;
		record.close();
		
		
	}

	
	
	printf("finished");
}

template<typename T>
void ndarraytoT(np::ndarray& source, T** destination)
{
	int rows = source.shape(0);
	int cols = source.shape(1);
	for (int i = 0; i < rows; i++) {
		//reinterpret_cast<double*>(destination.get_data())=source[i];
		//std::cout << "source["<<i<<"][0]="<< source[i][0]<<" ";
		//std::copy(source[i], source[i] + cols, reinterpret_cast<T*>(destination.get_data() + sizeof(T) * cols * i));
		std::copy(reinterpret_cast<T*>(source.get_data() + sizeof(T) * cols * i), reinterpret_cast<T*>(source.get_data() + sizeof(T) * cols * (i + 1)), destination[i]);
		//std::copy(source[i], source[i] + c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ols, reinterpret_cast<T*>(destination.get_data()));
	}
};

template<typename T>
void Ttondarray(T** source, np::ndarray& destination)
{
	int rows = destination.shape(0);
	int cols = destination.shape(1);
	for (int i = 0; i < rows; i++) {
		//reinterpret_cast<double*>(destination.get_data())=source[i];
		//std::cout << "source["<<i<<"][0]="<< source[i][0]<<" ";
		std::copy(source[i], source[i] + cols, reinterpret_cast<T*>(destination.get_data() + sizeof(T) * cols * i));
		//std::copy(reinterpret_cast<T*>(source.get_data() + sizeof(T) * cols * i), reinterpret_cast<T*>(source.get_data() + sizeof(T) * cols * (i + 1)), destination[i]);
		//std::copy(source[i], source[i] + c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ols, reinterpret_cast<T*>(destination.get_data()));
	}
};

void Encoder_mono(int height, int width, int img_num, int partition, np::ndarray & plain, np::ndarray & pred, np::ndarray & context, std::string filename) {
	char* chr_y;
	char base[40];
	std::string code(".bin"), code_y, base_new;
	size_t pos;
	pos = filename.find(code);
	code_y = filename.replace(pos, filename.length(), "_" + std::to_string(img_num) + "_" + std::to_string(partition) + ".bin");
	chr_y = _strdup(code_y.c_str());

	base_new = "data/base_" + std::to_string(img_num) + "_" + std::to_string(partition) + ".txt";
	strcpy(base, base_new.c_str());

	static float** P = (float**)calloc(height, sizeof(float*));
	static int** Y = (int**)calloc(height, sizeof(int*));
	static float** C = (float**)calloc(height, sizeof(float*));
	for (int r = 0; r < height; r++) {
		P[r] = (float*)calloc(width, sizeof(float));
		Y[r] = (int*)calloc(width, sizeof(int));
		C[r] = (float*)calloc(width, sizeof(float));
	}
	ndarraytoT<int>(plain, Y);
	//printf("converted Y, image %d, partition %d, first pixel %d\n", img_num, partition, Y[0][1]);
	//arraytotxt(base, height, width, &Y);
	ndarraytoT<float>(pred, P); 
	//printf("converted P, image %d, partition %d, first pixel %f\n", img_num, partition, P[0][0]);
	ndarraytoT<float>(context, C);
	//printf("converted C, image %d, partition %d, first pixel %f\n", img_num, partition, C[0][0]);


	float bpp = runEncoder_pixel_2(Y, chr_y, P, C, height, width);
	printf("encoded image %d, partition %d\n", img_num, partition);
	if (partition == 4) {
		std::ofstream record("record.txt", std::ios::app);
		record << std::endl;
		record.close();
	}
	
};

np::ndarray Decoder_mono(int height, int width, int img_num, int partition, np::ndarray& pred, np::ndarray& context, std::string filename) {
	char* chr_y;
	char out[40];
	std::string code(".bin"), code_y, out_new;
	size_t pos;
	pos = filename.find(code);
	code_y = filename.replace(pos, filename.length(), "_" + std::to_string(img_num) + "_" + std::to_string(partition) + ".bin");
	chr_y = _strdup(code_y.c_str());

	out_new = "data/out_" + std::to_string(img_num) + "_" + std::to_string(partition) + ".txt";
	strcpy(out, out_new.c_str());


	float** P = (float**)calloc(height, sizeof(float*));
	float** C = (float**)calloc(height, sizeof(float*));
	for (int r = 0; r < height; r++) {
		P[r] = (float*)calloc(width, sizeof(float));
		C[r] = (float*)calloc(width, sizeof(float));
	}
	ndarraytoT<float>(pred, P);
	ndarraytoT<float>(context, C);
	int** R = runDecoder_pixel_2(chr_y, P, C, out);
	//arraytotxt(out, height, width, &R);
	np::ndarray py_array = np::empty(nb::make_tuple(height, width), np::dtype::get_builtin<int>());
	Ttondarray<int>(R, py_array);

	free2D_f(P);
	free2D_f(C);

	return py_array;
};


np::ndarray boosttest(np::ndarray & np_array) {
	int rows = 10;
	int cols = 10;
	//prepare a new matrix for the output.
	static double** M = (double**)calloc(rows, sizeof(double*)); //allocate 4 rows
	for (int r = 0; r < rows; r++) {
		M[r] = (double*)calloc(cols, sizeof(double));
	}
	
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			M[r][c] = (double)(r * c);
		}
	}
	
	//float* arr = (float*)np_array.get_data();
	
	/*for (int i = 0; i < 10; i++) {
		record << arr[i] << std::endl;
		//arr[i] = arr[i] * 2;
		
	}*/
	
	//pass the target ndarray (existing, of same dimension)
	ndarraytoT<double>(np_array, M);
	
	std::ofstream record("record2.txt", std::ios::app);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			M[i][j] = 2 * M[i][j];
			record << M[i][j] << " ";
		}
		record << std::endl;
	}

	record.close();

	np::ndarray py_array = np::empty(nb::make_tuple(rows, cols), np::dtype::get_builtin<double>());
	Ttondarray<double>(M, py_array);
	/*
	np::ndarray py_array = np::from_data(M, np::dtype::get_builtin<double>(),
		nb::make_tuple(10,10),
		nb::make_tuple(10*sizeof(double), sizeof(double)),
		nb::object());
	*/
	
	return py_array;
}

BOOST_PYTHON_MODULE(ICIP_Compression) {
	np::initialize();
	boost::python::def("welcome", test);
	boost::python::def("arraytest", boosttest);
	boost::python::def("runencoder", Encoder_mono);
	boost::python::def("rundecoder", Decoder_mono);
}