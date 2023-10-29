#pragma once
void saveImage(const char* filename);

void createImg(int argc, char** argv, std::vector<std::vector<int>>& ivctOut
    , const int IRows, const int ICols, const char* filename);

void saveImage(const char* filename);

void vctImPrepare(const int IRows, const int ICols, int* piarrImOut
    , std::vector<std::vector<int>>& ivctOut);
