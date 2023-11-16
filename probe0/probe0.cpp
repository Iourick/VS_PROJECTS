#include "read_and_write_log.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


#ifdef _WIN32 // For Windows
#include <Windows.h>
#include <Lmcons.h> // Required for UNLEN constant
#else // For Linux
#include <unistd.h>
#include <limits.h>
#endif
#define _CRT_SECURE_NO_WARNINGS
using namespace std;


int main() {
    const char* filename = "info.log"; // File path and name

    // Sample data for writing into the file
    const char* projectName = "YourProjectName"; // Replace with your project's name
    const char* strProjectName = "YourStringProjectName"; // Replace with your string project name
    int quanthOfChunk = 5; // Sample length of chunk
    int lengthOfChunk = 128; // Sample length of chunk
    int successfulChunks[] = { 1, 2, 3, 4, 5 }; // Sample successful chunks
    int runningTime = 10; // Sample running time in seconds

    // Write information to the file
    fncWriteLog(filename, projectName, strProjectName, lengthOfChunk, quanthOfChunk, successfulChunks, runningTime);

    // Variables to store read information
    int readLengthOfChunk = 0;
    int lengthChunk = 0;
    int quantChunks = 0;
    int arrChunks[100] = { 0 }; // Adjust the size as per your requirements
    int reutime = 0;
    // Read information from the file
    int readStatus = fncReadLog(filename, &readLengthOfChunk,  &quantChunks, arrChunks);

    // Display the read information
    if (readStatus == 0) {
        std::cout << "Length of Chunk: " << readLengthOfChunk << std::endl;
        std::cout << "Total Chunks in File: " << quantChunks << std::endl;
        std::cout << "Numbers of Successful Chunks: [";
        for (int i = 0; i < readLengthOfChunk; ++i) {
            std::cout << arrChunks[i];
            if (i != readLengthOfChunk - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}
