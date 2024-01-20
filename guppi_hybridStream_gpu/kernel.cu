﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtCu0.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
#include "kernel.cuh"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"
#include "StreamParams.h"

#include "read_and_write_log.h"
#include "Constants.h"

#include <fstream>
#include "stdio.h"
#include <sstream>
#include "Session.cuh"
#include <limits.h>
#include <stdint.h>
#include "OutChunk.h"
#include "OutChunkHeader.h"



//#define _CRT_SECURE_NO_WARNINGS
using namespace std;

size_t free_bytes, total_bytes;
cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;


const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//blc20_guppi_57991_49905_DIAG_FRB121102_0011.0007.raw";
const char PAthOutFile[MAX_PATH_LENGTH] = "OutPutFile.bn";


const double VAlD_max = 1.5;
#define LENGTH_OF_PULSE 25.0E-8//25.0E-7//25.0E-8
const float SIgma_Bound = 10.;
// maximal length of summation window
#define MAX_LENGTH_SUMMATION_WND 10
#include <Windows.h>
int main()
{
//#include <iostream>
//
//#ifdef _WIN32 // Windows
//
//#include <Windows.h>
//
//    void emitSound(int frequency, int duration) {
//        Beep(frequency, duration);
//    }
//
//#else // Linux
//
//#include <cmath>
//#include <alsa/asoundlib.h>
//
//    void emitSound(int frequency, int duration) {
//        int rate = 44100; // Sampling rate
//        snd_pcm_t* handle;
//        snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
//        snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, rate, 1, 500000);
//
//        short buf[rate * duration];
//
//        for (int i = 0; i < rate * duration; i++) {
//            int sample = 32760 * sin(2 * M_PI * frequency * i / rate);
//            buf[i] = sample;
//        }
//
//        snd_pcm_writei(handle, buf, rate * duration);
//        snd_pcm_close(handle);
//    }
//
//#endif
//
//    int main() {
//        int frequency = 1500; // Frequency in hertz
//        int duration = 500;   // Duration in milliseconds
//
//        emitSound(frequency, duration);
//
//        return 0;
//    }

    CSession* pSession = new CSession(PAthGuppiFile, PAthOutFile, LENGTH_OF_PULSE, VAlD_max, SIgma_Bound, MAX_LENGTH_SUMMATION_WND);
    unsigned long long ilength = 0;
    int iBlocks = pSession->calcQuantRemainBlocks(&ilength);
                           
    pSession->launch();

    delete pSession;

    return 0;
}
