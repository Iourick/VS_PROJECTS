#include "StreamParams.h"
#include <thrust/complex.h>
#include <cufft.h>
//#include <complex>

using namespace std;

class CStreamParams;
int fncHybridDedispersionStream( int* piarrNumSuccessfulChunks, float* parrCoherent_d, int& quantOfSuccessfulChunks
	, CStreamParams* pStreamPars);

bool fncChunkHybridDedispersion_gpu(cufftComplex* pcmparrRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const int IMaxDT, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d
	, void* pAuxBuff_fdmt, void* pAuxBuff_the_rest, float* d_arrfdmt_norm);

