// headerRead0.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>


//#include <fstream>
//#include <string>
#include "stdio.h"
//#include <sstream>
//#include "MainStream.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace std;

int main()
{
    std::cout << LONG_MAX << endl;
    char file_name[] = "data.dat";
    

    // binary array
    float arr[20] = { 0 };
    for (int i = 0; i < 20; ++i)
    {
        arr[i] = (float)i;
    }

    

    // 2. add 10 spaces to the end
    // 
    char charr[] = "AAA =200.   VRD = 4000 \nghgh  dvdv\nququ = 5090\n            END";
    FILE* file3 = fopen(file_name, "wb");
    fwrite(charr, sizeof(char), sizeof(charr) / sizeof(char), file3);
    fclose(file3);
    
        FILE* file = fopen(file_name, "ab+"); // Open file in append and read/write mode for binary data

        if (file != NULL) {
            
            int numberOfSpacesToAdd = 10; // Change this value to the number of spaces you want to add

            // Move file pointer to the end of the file
            fseek(file, 0, SEEK_END);

            // Write spaces to the file
            for (int i = 0; i < numberOfSpacesToAdd; ++i) {
                fputc(' ', file); // Write a space character
            }

            fwrite(arr, sizeof(float), sizeof(arr) / sizeof(float), file);

            // Close the file
            fclose(file);
            printf("Added %d spaces to the end of the file.\n", numberOfSpacesToAdd);
        }
        else {
            printf("Failed to open the file.\n");
        }

        
    // 3
    // // read file in r mode
        FILE* file1 = fopen(file_name, "rb");
        char buffer[1000];
        fread(buffer, sizeof(char), 1000,file1);
        fclose(file1);
        std::cout << " buffer   ------------------- " << std::endl;
        std::cout << buffer << std::endl;

        char* pstr = strstr(buffer, "END");
        int pos0 = pstr - buffer;
    //m_wb_file = fopen(strOutPutPath, "wb");
    FILE* file2 = fopen(file_name, "rb");
    fseek(file2, pos0 + 4 +10,SEEK_SET);
    float arr1[20] = { 0. };
    fread(arr1, sizeof(float), 20, file2);

    fclose(file2);

    for (int i = 0; i < 20; ++i)
    {
        std::cout << arr1[i] << std::endl;
    }


    //int num_characters = 4096*4;  // Replace with the number of characters you want to read
    //FILE* file = NULL;
    //file = fopen(file_name, "r");
    //if (file == NULL) {
    //    perror("Error opening the file");
    //    return 1;
    //}

    //char buffer[4096*4];
    //fgets(buffer, sizeof(buffer), file);
    //std::cout << buffer << std::endl;
    //
    //char *pstr = strstr(buffer, "GBT");
    ////std::cout << pstr << std::endl;


   
    //fclose(file); // Close the file

    //std::cout << "pstr: " << pstr  << std::endl;  // Output the read content
    printf("Single quote: '\''\n");
    return 0;
}
