#ifndef __conv_h__
#define __conv_h__

enum class ConvType {
  CONV,
  CONVSHARED
};

void conv(unsigned char *N, float *F, unsigned char *P, int r, int height, int width);
void conv_shared(unsigned char *N, float *F, unsigned char *P, int r, int height, int width);

#endif