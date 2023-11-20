#include "StreamParams.h"
//#include <thrust/complex.h>
#include <complex>

using namespace std;

class CStreamParams;
int fncHybridDedispersionStream( int* piarrNumSuccessfulChunks, float* parrCoherent_d, int& quantOfSuccessfulChunks
	, CStreamParams* pStreamPars);

bool fncChunkHybridDedispersion(std::complex<float>* pRawSignalCur, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d);