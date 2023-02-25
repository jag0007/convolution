#ifndef __conv_h__
#define __conv_h__

// set up constant memory
// supports up to 32x32 filter
__constant__ float cF[25];

enum class ConvType {
  CONV,
  CONVSHARED,
  CONVCONST,
  CONVSHAREDTILE
};

void conv(unsigned char *N, float *F, unsigned char *P, int r, int height, int width);
void conv_shared(unsigned char *N, float *F, unsigned char *P, int r, int height, int width);
void conv_constant(unsigned char *N, float *F, unsigned char *P, int r, int height, int width);
void conv_shared_tile(unsigned char *N, float *F, unsigned char *P, int r, int height, int width);

#endif