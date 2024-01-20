#pragma once
#include "stdio.h" 
#include "GuppHeader.h"
#include <cufft.h>
#include <math.h>
#include "Constants.h"

#define MAX_PATH_LENGTH 1000

extern const unsigned long long TOtal_GPU_Bytes;
class CGuppHeader;

class CSession
{
public:
	~CSession();
	CSession();
	CSession(const  CSession& R);
	CSession& operator=(const CSession& R);
	CSession(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
	//--------------------------------------------------------	
	FILE* m_rbFile;
	FILE* m_wb_file;
	char m_strGuppiPath[MAX_PATH_LENGTH];
	char m_strOutPutPath[MAX_PATH_LENGTH];
	int m_numSubStream;	
	CGuppHeader m_header;
	float m_t_p;
	double m_d_max;
	float m_sigma_bound;
	int m_length_sum_wnd;
	//----------------------------------------------------------
	int calcQuantRemainBlocks(unsigned long long* pilength);

	int launch();

	void writeReport(int nS);

	inline int calc_len_sft(const float chanBW)
	{
		return (m_t_p > 2. / (chanBW * 1.0E6)) ? pow(2, ceil(log2(m_t_p * chanBW * 1.0E6))) : 1;
	}

	long long calcLenChunk(const int n_p);

};
