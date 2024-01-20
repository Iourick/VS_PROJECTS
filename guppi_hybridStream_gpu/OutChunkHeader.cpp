#include "OutChunkHeader.h"
COutChunkHeader::COutChunkHeader()
{
	m_nrows = 0;
	m_ncols = 0;
	m_nSucessRow = 0;
	m_nSucessCol = 0;
	m_SNR = 0;
	m_coherentDedisp = 0.;
	m_numBlock = 0;
	m_numChunk = 0;
	m_wnd_width = 0.;
}
//-----------------------------------------------------------

COutChunkHeader::COutChunkHeader(const  COutChunkHeader& R)
{
	m_nrows = R.m_nrows;
	m_ncols = R.m_ncols;
	m_nSucessRow = R.m_nSucessRow;
	m_nSucessCol = R.m_nSucessCol;
	m_SNR = R.m_SNR;
	m_coherentDedisp = R.m_coherentDedisp;
	m_numBlock = R.m_numBlock;
	m_numChunk = R.m_numChunk;
	m_wnd_width = R.m_wnd_width;
}
//-------------------------------------------------------------------

COutChunkHeader& COutChunkHeader::operator=(const COutChunkHeader& R)
{
	if (this == &R)
	{
		return *this;
	}

	m_nrows = R.m_nrows;
	m_ncols = R.m_ncols;
	m_nSucessRow = R.m_nSucessRow;
	m_nSucessCol = R.m_nSucessCol;
	m_SNR = R.m_SNR;
	m_coherentDedisp = R.m_coherentDedisp;
	m_numBlock = R.m_numBlock;
	m_numChunk = R.m_numChunk;
	m_wnd_width = R.m_wnd_width;
	return *this;
}
//------------------------------------------------------------------
COutChunkHeader::COutChunkHeader(
	const int nrows
	, const int ncols
	, const int nSucessRow
	, const int nSucessCol
	, const int wnd_width
	, const float SNR
	, const float 	coherentDedisp
	, const int numBlock
	, const int numChunk
)

{
	m_nrows = nrows;

	m_ncols = ncols;

	m_nSucessRow = nSucessRow;

	m_nSucessCol = nSucessCol;

	m_SNR = SNR;

	m_coherentDedisp = coherentDedisp;

	m_numBlock = numBlock;

	m_numChunk = numChunk;

	m_wnd_width = wnd_width;

}

