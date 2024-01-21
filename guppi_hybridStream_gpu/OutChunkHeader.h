#pragma once
class COutChunkHeader
{
public:
	COutChunkHeader();
	COutChunkHeader(const  COutChunkHeader& R);
	COutChunkHeader& operator=(const COutChunkHeader& R);
	COutChunkHeader(
		const int nrows
		, const int ncols
		, const int nSucessRow
		, const int nSucessCol
		, const int width
		, const float SNR
		, const float 	coherentDedisp
		, const int numBlock
		, const int numChunk
	);
	int m_nrows;
	int m_ncols;
	int m_nSucessRow;
	int m_nSucessCol;
	float m_SNR;
	float m_coherentDedisp;
	int m_numBlock;
	int m_numChunk;
	int m_wnd_width;

	void createOutStr(char* pstr);

};

