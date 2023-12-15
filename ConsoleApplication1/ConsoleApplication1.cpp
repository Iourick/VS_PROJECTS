// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char* argv[]) {
    printf("Number of arguments: %d\n", argc);

    printf("Argument values:\n");
    for (int i = 0; i < argc; ++i) {
        printf("Argument %d: %s\n", i, argv[i]);
    }

    char atr[100] = { 0 };
    strcpy(atr, argv[1]);
    int it = atoi(atr);
    printf("Argument %d\n", it);
    return 0;
}

