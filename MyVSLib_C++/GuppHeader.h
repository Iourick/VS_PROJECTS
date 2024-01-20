#pragma once
#include <stdio.h>
enum EN_telescope{ GBT,PARKES };
class CGuppHeader
{
	public:
		CGuppHeader();
		CGuppHeader(const  CGuppHeader& R);
		CGuppHeader&operator=(const CGuppHeader& R);
		CGuppHeader(
			const int nbits
			, const float chanBW
			, const int npol
			, const bool bdirectIO
			, const float centfreq
			, const int nchan
			, const float obsBW
			, const int nblocksize
			, const  EN_telescope TELESCOP
			//, const int npkt
		);
/************************************************************************************************************/
		// NBITS: Number of bits in each complex component per
		//sample—one complex - valued sample has 2 x NBITS bits.
		int m_nbits;
		// CHAN_BW: Bandwidth (in MHz) of a single channel. Will be
		//negative if the frequency axis is reversed(e.g., due to a
		//lower sideband mix).
		float m_chanBW;
		// NPOL: Number of samples per time step—4 corresponds to
		 //dual - polarization complex data.
		int m_npol;
		//DIRECTIO: Indicates whether Direct I/O was used to
		//optimize file writing.If used, indicated by a non - zero
		//value, the header is padded to be a multiple of 512 bytes.
		bool m_bdirectIO;
	    //OBSFREQ: Center frequency of the frequencies spanned by the
		//data in the RAW file.
		float m_centfreq;
		//The number of frequency channels contained
		//within the file (=OBSNCHAN).
		int m_nchan;
	    // OBSBW: Width of passband in this node; negative sign
		// indicates reversed frequency axis.
		float m_obsBW;
	    // BLOCSIZE: The size of the following raw data segment, in
		//bytes.Does not include padding bytes from Direct I / O.Can
		//be expressed as 2 x NPOL x NTIME x NCHAN x NBITS /8
		long long  m_nblocksize;
	    //TELESCOP: Telescope name.
	    // NPKT: Number of packets received for this block.
		//int m_npkt;

		EN_telescope m_TELESCOP;

/**********************************************************************************************/
		static bool readHeader(FILE* r_file
		, int* nbits
		, float* chanBW
		, int* npol
		, bool* bdirectIO
		, float* centfreq
		, int* nchan
		, float* obsBW
		, long long* nblocksize
		, EN_telescope* TELESCOP		
		);
		
};

