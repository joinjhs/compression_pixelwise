#pragma once
#include "arithmetic_codec.h"
#include "utils.h"

void initModel_y(Adaptive_Data_Model dm[]);

void initModel_uv(Adaptive_Data_Model dm[]);

int calcContext(float ctx, int channel);

void runNetwork(struct stNeuralNetwork *pNN, WEIGHT_TYPE *in, float *pred, float *ctx, float *hidden);

float runEncoder(char *infile, char *codefile_y, char *codefile_u, char *codefile_v, char *weight_y, char *weight_u, char *weight_v);

float runEncoder_pixel(char* infile, char* codefile, char* pred, char* context, int height, int width);

float runEncoder_pixel_2(int** Y, char* codefile, float** P, float** C, int height, int width);