#include "main.h"
#include <iostream>
//#include <complex>
#include <fftw3.h>
#include <chrono>
#include <cmath>
#include "math.h"

const double M_PI_ = 3.1415926;

#define _CRT_SECURE_NO_WARNINGS
using namespace std;
char* global_wisdom_str = NULL;
fftwf_plan create_plan(const unsigned int size)
{
    fftwf_complex* in = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);
    fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);

    // Create the plan
    fftwf_plan plan = fftwf_plan_dft_1d(size, in, out, FFTW_FORWARD, /*FFTW_ESTIMATE*/ FFTW_MEASURE);
    fftw_export_wisdom_to_filename("wisdom.wis");

    // Export the plan's wisdom to a string
    //global_wisdom_str = fftw_export_wisdom_to_string();
    global_wisdom_str = fftw_export_wisdom_to_string();
    // Destroy the plan when no longer needed
    //fftw_destroy_plan(plan);

    // Free allocated memory
    fftwf_free(in);
    fftwf_free(out);
    return plan;
}

int main(int argc, char** argv)
{  
    
    fftwf_cleanup();
    int size =  2 << 20; // Size of your 1D array

    fftwf_plan plan = nullptr;

        // Create a plan for a 1D FFT
    fftwf_complex* in1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);
    fftwf_complex* out1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);

    fftwf_complex* in2 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);
    fftwf_complex* out2 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);
    
    for (int i = 0; i < size; ++i)
    {
        in1[i][0] = sin((float)i / 10. / M_PI_);
        in1[i][1] = 0.;
        in2[i][0] = 0.;
        in2[i][1] = sin((float)i / 10. / M_PI_);

    }
    //--------------------------------------------------------------------------------------
    //------  1. measurement for simple code --------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------

    int num = 100;
    auto start = std::chrono::high_resolution_clock::now();
    int icur = 1;
    for (int i = 0; i < num; ++i)
    {
        if (icur > 0)
        {
            plan = fftwf_plan_dft_1d(size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
        }
        else
        {
            plan = fftwf_plan_dft_1d(size, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
        }

        // Use the plan to execute the FFT with input and output arrays in1 and out1
        fftwf_execute(plan);
        // Destroy the plan when it's no longer needed
        fftwf_destroy_plan(plan);
        icur = -icur;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "1. Time taken by simple case " << duration.count() / ((double)num) << " microseconds" << std::endl;
    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    
//--------------------------------------------------------------------------------------
//------  2. measurement plan was not destroyed --------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------       
    fftwf_cleanup();
    start = std::chrono::high_resolution_clock::now();
    icur = 1;
    for (int i = 0; i < num; ++i)
    {
        if (icur > 0)
        {
            plan = fftwf_plan_dft_1d(size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
        }
        else
        {
            plan = fftwf_plan_dft_1d(size, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
        }
        fftwf_execute(plan);
        icur = -icur;
    }
    fftwf_destroy_plan(plan);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "2. Time taken by not destroyed plan: " << duration.count() / ((double)num) << " microseconds" << std::endl;
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//------  3. measurement fftw_import_wisdom_from_filename --------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------   
    fftwf_cleanup();
    create_plan(size);
    //fftw_import_wisdom_from_filename("wisdom.wis");
    fftw_import_wisdom_from_string(global_wisdom_str);
    plan = fftwf_plan_dft_1d(size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE /*FFTW_WISDOM_ONLY*/);;
    start = std::chrono::high_resolution_clock::now();
    icur = 1;
    for (int i = 0; i < num; ++i)
    {
        if (icur > 0)
        {
            fftwf_execute_dft(plan, in1, out1);
        }
        else
        {
            fftwf_execute_dft(plan, in2, out2);
        }
        
        
        icur = -icur;
        

    }
    fftwf_destroy_plan(plan);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "3. Time taken by fftw_import_wisdom_from_filename: " << duration.count() / ((double)num) << " microseconds" << std::endl;
    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //------  4. measurement fftw_import_wisdom_from_file --------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------   
   
    //fftwf_cleanup();
    //create_plan(size);
    //auto start1 = std::chrono::high_resolution_clock::now();
    //FILE* wisdomfile = fopen("wisdom.wis", "rb");
    //
    //fftw_import_wisdom_from_file(wisdomfile);
    //
    //plan = fftwf_plan_dft_1d(size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE /*FFTW_WISDOM_ONLY*/);;
    //for (int i = 0; i < num; ++i)
    //{
    //    fftwf_execute_dft(plan, in1, out1);
    //    //plan = fftwf_plan_dft_1d(size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE /*FFTW_WISDOM_ONLY*/);;

    //    // Use the saved_plan for FFT computations
    //    if (plan == NULL) {
    //        printf("Error: Failed to retrieve the plan from wisdom file.\n");
    //        // Handle the error accordingly
    //    }
    //    //fftwf_execute(plan);

    //}
    //fftwf_destroy_plan(plan);
    //fclose(wisdomfile);
    
    //auto end1 = std::chrono::high_resolution_clock::now();
    //auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    //std::cout << "4. Time taken by not fftw_import_wisdom_from_file: " << duration1.count() / ((double)num) << " microseconds" << std::endl;

    //--------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //------  5. measurement fftw_import_wisdom_from_string --------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------   
    fftwf_cleanup();
    create_plan(size);

    start = std::chrono::high_resolution_clock::now();
    fftw_import_wisdom_from_string(global_wisdom_str);
    icur = 1;
    plan = fftwf_plan_dft_1d(size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE /*FFTW_WISDOM_ONLY*/);;
    for (int i = 0; i < num; ++i)
    {
         if (icur > 0)
        {
            fftwf_execute_dft(plan, in1, out1);
        }
        else
        {
            fftwf_execute_dft(plan, in2, out2);
        }


        icur = -icur;
        

    }
    fftwf_destroy_plan(plan);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "5. Time taken by not fftw_import_wisdom_from_string: " << duration.count() / ((double)num) << " microseconds" << std::endl;
    

    // Free allocated memory
    fftwf_free(in1);
    fftwf_free(out1);
    fftwf_free(in2);
    fftwf_free(out2);
    
    return 0;    
}
