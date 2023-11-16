#pragma once
int fncReadLog(const char* filename, int* lengthOfChunk, int* quantChunks, int* arrChunks);

void fncWriteLog(const char* filename, const char* projectName, const char* strProjectName, int lengthOfChunk
    , int quantOfChunk, const int* successfulChunks, int runningTime);
