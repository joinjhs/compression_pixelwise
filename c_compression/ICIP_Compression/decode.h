#pragma once

float runDecoder(char *codefile_y, char *codefile_u, char *codefiel_v, char *outfile, char *weight_y, char *weight_u, char *weight_v);
void runDecoder_pixel(char* outfile, char* codefile, char* pred, char* context);
int** runDecoder_pixel_2(char* codefile, float** P, float** C, char* outfile);