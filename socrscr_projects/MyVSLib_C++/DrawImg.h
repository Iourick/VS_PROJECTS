#pragma once
#include <vector>
using namespace std;
void saveImage(const char* filename);

void createImg(int argc, char** argv, int* piarrImOut, const int IRows, const int ICols,const char* filename);

void saveImage(const char* filename);

void createImg_(int argc, char** argv, std::vector<int>& vctOut1D
    , const int IRows, const int ICols, const char* filename);
