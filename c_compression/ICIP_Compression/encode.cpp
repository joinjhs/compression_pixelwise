#include "encode.h"
#include "stdio.h"
#include "assert.h"
#include <windows.h>
#include <string>
#include <time.h>
#include "ppm_io.h"
#include <fstream>

#pragma warning(disable: 4996)

void initModel_y(Adaptive_Data_Model dm[]) {
	for (int i = 0; i < NUM_CTX; i++) {
		dm[i].set_alphabet(ALPHABET_MAX);

		if (i == 0)
			dm[i].set_distribution(0.7f);
		else
			dm[i].set_distribution(0.8f);
	}
}

void initModel_uv(Adaptive_Data_Model dm[]) {
	for (int i = 0; i < NUM_CTX; i++) {
		dm[i].set_alphabet(ALPHABET_MAX + 1);

		if (i == 0)
			dm[i].set_distribution(0.7f);
		else
			dm[i].set_distribution(0.8f);
	}
}

void runNetwork(struct stNeuralNetwork *pNN, WEIGHT_TYPE *in, float *pred, float *ctx, float *hidden) {

	WEIGHT_TYPE *L1 = new WEIGHT_TYPE[pNN->n_hidden];
	WEIGHT_TYPE *L2 = new WEIGHT_TYPE[pNN->n_hidden];
	WEIGHT_TYPE *Lout = new WEIGHT_TYPE[pNN->n_hidden];

	matMul(in, &pNN->Win[0][0], L2, pNN->n_in, pNN->n_hidden);
	matAdd(L2, pNN->B[0], L2, pNN->n_hidden);
	relu(L2, pNN->n_hidden);

	for (int i = 0; i < pNN->n_layer - 2; i++) {
		matMul(L2, &pNN->W[i][0][0], L1, pNN->n_hidden, pNN->n_hidden);
		matAdd(L1, pNN->B[i + 1], L2, pNN->n_hidden);
		relu(L2, pNN->n_hidden);
	}

	matMul(L2, &pNN->Wout[0][0], Lout, pNN->n_hidden, pNN->n_out);
	matAdd(Lout, pNN->Bout, Lout, pNN->n_out);

	*pred = (float)Lout[0];
	*ctx = MAX((float)Lout[1], 0.0);

	for (int i = 0; i < pNN->n_hidden; i++) {
		hidden[i] = (float)L2[i];
	}

	free(L1);
	free(L2);
	free(Lout);
}

int calcContext(float ctx, int channel) {
	// Channel : Y - 0, U - 1, V - 2, Original - 3

	float *TH;

	if (channel == 0) {
		float TH_Y[] = { 0.0, 0.0, 0.026, 0.501, 0.506, 0.508, 0.511, 0.516, 0.541, 0.692, 0.781, 0.876, 0.962, 0.971, 1.073, 1.193, 1.319, 1.464, 1.622, 1.794, 1.985, 2.196, 2.438, 2.715, 3.043, 3.441, 3.943, 4.608, 5.548, 7.006, 9.803 };
		TH = TH_Y;
	}
	else if (channel == 1) {
		float TH_U[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0102, 0.4978, 0.4978, 0.4978, 0.4988, 0.4988, 0.4988, 0.5035, 0.572, 0.646, 0.703, 0.765, 0.823, 0.882, 0.943, 1.017, 1.1148, 1.239, 1.405, 1.637, 2.0294, 2.7805 };
		TH = TH_U;
	}
	else if (channel == 2) {
		float TH_V[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.105, 0.265, 0.443, 0.491, 0.500, 0.511, 0.564, 0.670, 0.802, 0.905, 0.924, 0.931, 0.960, 1.003, 1.063, 1.137, 1.229, 1.349, 1.509, 1.737, 2.104, 2.863 };
		TH = TH_V;
	}
	else {
		float TH_common[] = { .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14 };
		TH = TH_common;
	}



	int i;

	for (i = 0; i < NUM_CTX; i++) {
		if (ctx < TH[i]) break;
	}

	return MIN(i, NUM_CTX - 1);
}

int makeSymbol_y(int X, int left, float pred) {

	int sym;
	float Pd = pred + left;
	int P = UINT8(Pd);

	if (P >= 128) {
		Pd = 255 - Pd;
		P = 255 - P;
		X = 255 - X;
	}

	if (Pd > P) {
		if (X <= P)
			sym = 2 * (P - X);
		else if (X > 2 * P)
			sym = X;
		else
			sym = 2 * (X - P) - 1;
	}
	else {
		if (X < P)
			sym = 2 * (P - X) - 1;
		else if (X > 2 * P)
			sym = X;
		else
			sym = 2 * (X - P);
	}

	return sym;
}

int makeSymbol_uv(int X, int left, float pred) {

	int sym;
	float Pd = pred + left;
	int P = int(Pd);

	if (P >= 256) {
		Pd = 511 - Pd;
		P = 511 - P;
		X = 511 - X;
	}

	if (Pd > P) {
		if (X <= P)
			sym = 2 * (P - X);
		else if (X > 2 * P)
			sym = X;
		else
			sym = 2 * (X - P) - 1;
	}
	else {
		if (X < P)
			sym = 2 * (P - X) - 1;
		else if (X > 2 * P)
			sym = X;
		else
			sym = 2 * (X - P);
	}

	return sym;
}

void encodeMag_y(int mag, Arithmetic_Codec *pCoder, Adaptive_Data_Model *pDm) {
	assert(SYMBOL_MAX == 16);

	if (mag < SYMBOL_MAX) {
		pCoder->encode(mag, *pDm);
	}
	else if (mag < 32) {
		pCoder->encode(SYMBOL_MAX, *pDm);
		pCoder->put_bits((mag - 16), 4);
	}
	else if (mag < 64) {
		pCoder->encode(SYMBOL_MAX + 1, *pDm);
		pCoder->put_bits((mag - 32), 5);
	}
	else if (mag < 128) {
		pCoder->encode(SYMBOL_MAX + 2, *pDm);
		pCoder->put_bits((mag - 64), 6);
	}
	else {
		pCoder->encode(SYMBOL_MAX + 3, *pDm);
		pCoder->put_bits((mag - 128), 7);
	}
}

void encodeMag_uv(int mag, Arithmetic_Codec *pCoder, Adaptive_Data_Model *pDm) {
	assert(SYMBOL_MAX == 16);

	if (mag < SYMBOL_MAX) {
		pCoder->encode(mag, *pDm);
	}
	else if (mag < 32) {
		pCoder->encode(SYMBOL_MAX, *pDm);
		pCoder->put_bits((mag - 16), 4);
	}
	else if (mag < 64) {
		pCoder->encode(SYMBOL_MAX + 1, *pDm);
		pCoder->put_bits((mag - 32), 5);
	}
	else if (mag < 128) {
		pCoder->encode(SYMBOL_MAX + 2, *pDm);
		pCoder->put_bits((mag - 64), 6);
	}
	else if (mag < 256) {
		pCoder->encode(SYMBOL_MAX + 3, *pDm);
		pCoder->put_bits((mag - 128), 7);
	}
	else {
		pCoder->encode(SYMBOL_MAX + 4, *pDm);
		pCoder->put_bits((mag - 256), 8);
	}
}

float encode(FILE *fp_y, FILE *fp_u, FILE *fp_v, struct stNeuralNetwork *pNN_y, struct stNeuralNetwork *pNN_u, struct stNeuralNetwork *pNN_v, int **Y, int **U, int **V, int height, int width) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			U[y][x] += 255;
			V[y][x] += 255;
		}
	}

	Arithmetic_Codec coder[3];
	Adaptive_Data_Model dm[3][NUM_CTX];
	int ctxCnt[3][NUM_CTX] = { 0 };

	int y, x, i;
	int numPix = 0;

	int ctx_left = pNN_y->ctx_left;
	int ctx_up = pNN_y->ctx_up;
	int ctx_total = (ctx_left * 2 + 1)*ctx_up + ctx_left - 1;
	int hidden_unit = pNN_y->n_hidden;

	WEIGHT_TYPE nbr_y[3 * 11];

	initModel_y(dm[0]);
	initModel_uv(dm[1]);
	initModel_uv(dm[2]);

	int size = width * height;

	for (int i = 0; i < 3; i++) {
		coder[i].set_buffer(size);
		coder[i].start_encoder();
	}

	assert((unsigned short)width == width);
	assert((unsigned short)height == height);

	coder[0].put_bits((unsigned short)width, 16);
	coder[0].put_bits((unsigned short)height, 16);

	for (y = 0; y < ctx_up; y++) {
		for (x = 0; x < width; x++) {
			coder[0].put_bits(Y[y][x], 8);
			coder[1].put_bits(U[y][x], 9);
			coder[2].put_bits(V[y][x], 9);
			numPix++;
		}
	}

	for (y = ctx_up; y < height; y++) {
		for (x = 0; x < ctx_left; x++) {
			coder[0].put_bits(Y[y][x], 8);
			coder[1].put_bits(U[y][x], 9);
			coder[2].put_bits(V[y][x], 9);
			numPix++;;
		}

		for (x = ctx_left; x < width - ctx_left; x++) {

			int X_y = Y[y][x];
			int X_u = U[y][x];
			int X_v = V[y][x];

			int left_y = Y[y][x - 1];
			int left_u = U[y][x - 1];
			int left_v = V[y][x - 1];

			int cnt = 0;

			for (int i = 0; i < ctx_up + 1; i++) {
				for (int j = 0; j < 2 * ctx_left + 1; j++) {

					if (i == ctx_up && j >= ctx_left - 1)
						break;

					nbr_y[cnt] = Y[y - (ctx_up - i)][x - (ctx_left - j)];
					nbr_y[cnt + ctx_total] = U[y - (ctx_up - i)][x - (ctx_left - j)];
					nbr_y[cnt + 2 * ctx_total] = V[y - (ctx_up - i)][x - (ctx_left - j)];

					cnt++;
				}
			}

			for (i = 0; i < ctx_total; i++) {
				nbr_y[i] -= left_y;
				nbr_y[i + ctx_total] -= left_u;
				nbr_y[i + 2 * ctx_total] -= left_v;
			}

			float pred_y, ctx_y, pred_u, ctx_u, pred_v, ctx_v;
			float hidden_y[64], hidden_u[64], hidden_v[64];

			WEIGHT_TYPE input_u[64 + 33 + 2];
			WEIGHT_TYPE input_v[64 + 33 + 4];

			runNetwork(pNN_y, nbr_y, &pred_y, &ctx_y, hidden_y);

			for (int i = 0; i < hidden_unit + 2 + 3 * ctx_total; i++) {
				if (i < hidden_unit)
					input_u[i] = hidden_y[i];
				else if (i < hidden_unit + 3 * ctx_total)
					input_u[i] = nbr_y[i - hidden_unit];
				else if (i == hidden_unit + 3 * ctx_total)
					input_u[i] = X_y - left_y;
				else
					input_u[i] = pred_y;
			}

			runNetwork(pNN_u, input_u, &pred_u, &ctx_u, hidden_u);

			for (int i = 0; i < hidden_unit + 4 + 3 * ctx_total; i++) {
				if (i < hidden_unit)
					input_v[i] = hidden_u[i];
				else if (i < hidden_unit + 3 * ctx_total)
					input_v[i] = nbr_y[i - hidden_unit];
				else if (i == hidden_unit + 3 * ctx_total)
					input_v[i] = X_y - left_y;
				else if (i == hidden_unit + 3 * ctx_total + 1)
					input_v[i] = pred_y;
				else if (i == hidden_unit + 3 * ctx_total + 2)
					input_v[i] = X_u - left_u;
				else
					input_v[i] = pred_u;
			}

			runNetwork(pNN_v, input_v, &pred_v, &ctx_v, hidden_v);

			int iCtx_y = calcContext(ctx_y, -1);
			int iCtx_u = calcContext(ctx_u, -1);
			int iCtx_v = calcContext(ctx_v, -1);

			ctxCnt[0][iCtx_y]++;
			ctxCnt[1][iCtx_u]++;
			ctxCnt[2][iCtx_v]++;

			int sym_y = makeSymbol_y(X_y, left_y, pred_y);
			int sym_u = makeSymbol_uv(X_u, left_u, pred_u);
			int sym_v = makeSymbol_uv(X_v, left_v, pred_v);

			encodeMag_y(sym_y, &coder[0], &dm[0][iCtx_y]);
			encodeMag_uv(sym_u, &coder[1], &dm[1][iCtx_u]);
			encodeMag_uv(sym_v, &coder[2], &dm[2][iCtx_v]);

			numPix++;
		}

		for (x = width - pNN_y->ctx_left; x < width; x++) {
			coder[0].put_bits(Y[y][x], 8);
			coder[1].put_bits(U[y][x], 9);
			coder[2].put_bits(V[y][x], 9);
			numPix++;
		}
	}

	int bytes_y = coder[0].write_to_file(fp_y);
	int bytes_u = coder[1].write_to_file(fp_u);
	int bytes_v = coder[2].write_to_file(fp_v);

	int bytes = bytes_y + bytes_u + bytes_v;

	printf("%7.4f bpp (%d bytes)\n", 8.0*bytes / numPix, bytes);

	return bytes;
}

float encode_pixel(FILE *fp_y, int **Y, float**P, float**Ctx, int height, int width) {


	Arithmetic_Codec coder[1];
	Adaptive_Data_Model dm[1][NUM_CTX];
	int ctxCnt[1][NUM_CTX] = { 0 };

	int y, x, i;
	int numPix = 0;

	initModel_y(dm[0]);

	int size = width * height;

	coder[0].set_buffer(size);
	coder[0].start_encoder();

	assert((unsigned short)width == width);
	assert((unsigned short)height == height);

	coder[0].put_bits((unsigned short)width, 16);
	coder[0].put_bits((unsigned short)height, 16);


	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {

			int X_y = Y[y][x];
			int iCtx_y = calcContext(Ctx[y][x],-1);
			float pred_y = P[y][x];

			ctxCnt[0][iCtx_y]++;

			int sym_y = makeSymbol_y(X_y, 0, pred_y);

			encodeMag_y(sym_y, &coder[0], &dm[0][iCtx_y]);

			numPix++;
		}

	}

	int bytes_y = coder[0].write_to_file(fp_y);

	int bytes = bytes_y;

	//printf("%7.4f bpp (%d bytes)\n", 8.0*bytes / numPix, bytes);
	//printf("%7.4f ", 8.0*bytes / numPix);

	std::ofstream record("record.txt", std::ios::app);
	record << 8.0 * bytes / numPix << " ";
	record.close();

	return bytes;
}


float runEncoder_pixel(char *infile, char *codefile, char *pred, char *context, int height, int width) {

	FILE *fp_y;

	float** C = { 0 };
	int** Y = { 0 };
	float** P = { 0 };
	clock_t start = clock();
	txttoarray_f(context, height, width, &C);
	clock_t end = clock();
	//printf("reading txt: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	txttoarray(infile, height, width, &Y);
	txttoarray_f(pred, height, width, &P);

	//printf("Image : %s\n", infile);

	if (!(fp_y = fopen(codefile, "wb"))) {
		fprintf(stderr, "Code file open error(encoding).\n");
		exit(-1);
	}

	start = clock();
	int bytes = encode_pixel(fp_y, Y, P, C, height, width);
	end = clock();
	//printf("Encoding time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	//printf("%lf ", (double)(end - start) / CLOCKS_PER_SEC);
	std::ofstream record("record.txt", std::ios::app);
	record << (double)(end - start) / CLOCKS_PER_SEC << " ";
	record.close();
	fclose(fp_y);

	free2D(Y);
	free2D_f(C);
	free2D_f(P);


	return float(8.0*bytes / (width*height));
}

float runEncoder_pixel_2(int** Y, char* codefile, float** P, float** C, int height, int width) {

	FILE* fp_y;


	if (!(fp_y = fopen(codefile, "wb"))) {
		fprintf(stderr, "Code file open error(encoding).\n");
		exit(-1);
	}

	clock_t start = clock();
	int bytes = encode_pixel(fp_y, Y, P, C, height, width);
	clock_t end = clock();
	//printf("Encoding time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	//printf("%lf ", (double)(end - start) / CLOCKS_PER_SEC);
	std::ofstream record("record.txt", std::ios::app);
	record << (double)(end - start) / CLOCKS_PER_SEC << " ";
	record.close();
	fclose(fp_y);
	/*
	free2D(Y);
	free2D_f(C);
	free2D_f(P);
	*/

	return float(8.0 * bytes / (width * height));
}

float runEncoder(char *infile, char *codefile_y, char *codefile_u, char *codefile_v, char *weight_y, char *weight_u, char *weight_v) {

	FILE *fp_y, *fp_u, *fp_v;

	struct stNeuralNetwork NN_y, NN_u, NN_v;
	readWeight(weight_y, &NN_y);
	readWeight(weight_u, &NN_u);
	readWeight(weight_v, &NN_v);

	int **R, **G, **B;
	int **Y, **U, **V;
	int height, width, bitdepth;

	readPPM(infile, &R, &G, &B, &height, &width, &bitdepth);
	RGB2YUV(&R, &G, &B, &Y, &U, &V, &height, &width);;

	printf("Image : %s\n", infile);

	if (!(fp_y = fopen(codefile_y, "wb"))) {
		fprintf(stderr, "Code file open error(encoding).\n");
		exit(-1);
	}

	if (!(fp_u = fopen(codefile_u, "wb"))) {
		fprintf(stderr, "Code file open error(encoding).\n");
		exit(-1);
	}

	if (!(fp_v = fopen(codefile_v, "wb"))) {
		fprintf(stderr, "Code file open error(encoding).\n");
		exit(-1);
	}

	int bytes = encode(fp_y, fp_u, fp_v, &NN_y, &NN_u, &NN_v, Y, U, V, height, width);

	fclose(fp_y);
	fclose(fp_u);
	fclose(fp_v);

	free2D(R);
	free2D(G);
	free2D(B);
	free2D(Y);
	free2D(U);
	free2D(V);

	free2Dweight(NN_y.Win);
	free2Dweight(NN_y.Wout);
	free(NN_y.Bout);
	free(NN_y.B[0]);

	for (int i = 0; i < NN_y.n_layer - 2; i++) {
		free2Dweight(NN_y.W[i]);
		free(NN_y.B[i + 1]);
	}

	free2Dweight(NN_u.Win);
	free2Dweight(NN_u.Wout);
	free(NN_u.Bout);
	free(NN_u.B[0]);

	free2Dweight(NN_v.Win);
	free2Dweight(NN_v.Wout);
	free(NN_v.Bout);
	free(NN_v.B[0]);

	for (int i = 0; i < NN_u.n_layer - 2; i++) {
		free2Dweight(NN_u.W[i]);
		free(NN_u.B[i + 1]);
	}

	for (int i = 0; i < NN_v.n_layer - 2; i++) {
		free2Dweight(NN_v.W[i]);
		free(NN_v.B[i + 1]);
	}

	return float(8.0*bytes / (width*height));
}