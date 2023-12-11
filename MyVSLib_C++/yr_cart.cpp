#include "yr_cart.h"

#include  <cfloat>

//------------------------------------------------
void  findMaxMinOfArray(float* d_arrfdmt_norm, unsigned const int LEn, float* valmax, float* valmin
	, unsigned int* iargmax, unsigned int* iargmin)
{
	*valmax = -FLT_MAX;
	*valmin = FLT_MAX;
	*iargmax = -1;
	*iargmin = -1;

	for (int i = 0; i < LEn; ++i)
	{
		if (d_arrfdmt_norm[i] > (*valmax))
		{
			*valmax = d_arrfdmt_norm[i];
			*iargmax = i;
		}
		if (d_arrfdmt_norm[i] < (*valmin))
		{
			*valmin = d_arrfdmt_norm[i];
			*iargmin = i;

		}
	}
}
