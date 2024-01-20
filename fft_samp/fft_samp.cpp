#include <iostream>
#include <complex>
#include <fftw3.h>
#include <GL/glut.h>

#include <fstream>
#define _CRT_SECURE_NO_WARNINGS

void display() {
    glClear(GL_COLOR_BUFFER_BIT); // ������� ������ �����

    glBegin(GL_TRIANGLES); // ������ ��������� ������������
    glColor3f(1.0, 0.0, 0.0); // ��������� ����� (�������)
    glVertex2f(0.0, 0.5); // ������� 1
    glColor3f(0.0, 1.0, 0.0); // ��������� ����� (�������)
    glVertex2f(-0.5, -0.5); // ������� 2
    glColor3f(0.0, 0.0, 1.0); // ��������� ����� (�����)
    glVertex2f(0.5, -0.5); // ������� 3
    glEnd(); // ���������� ��������� ������������

    glFlush(); // �������� ������� �� �����
}
int main(int argc, char** argv)
{


    
        // Open the binary file for reading
        FILE* file = fopen("D://MyVSprojPy//hybrid//info.bin", "rb");
        if (file == nullptr) {
            std::cerr << "Error opening file." << std::endl;
            return 1;
        }

        // Variables to store data
        int i0, i1;
        float f0, f1, f2, f3;

        // Read the integer variables
        fread(&i0, sizeof(int), 1, file);
        fread(&i1, sizeof(int), 1, file);

        // Read the float variables
        fread(&f0, sizeof(float), 1, file);
        fread(&f1, sizeof(float), 1, file);
        fread(&f2, sizeof(float), 1, file);
        fread(&f3, sizeof(float), 1, file);

        // Read the complex array size
        

        // Allocate an array for the complex numbers
        std::complex<float>* array = new std::complex<float>[i0];

        
        fread(array, sizeof(std::complex<float>), i0, file);

        // Close the file
        fclose(file);

        // Use the variables as needed
        // ...

        // Don't forget to delete the array when you're done with it
            delete[] array;

        

    int N = 8; // Number of data points
    std::complex<float>* in = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);
    fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(in);
    fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * N);

    // Initialize 'in' with complex numbers using std::complex<float>
    for (int i = 0; i < N; ++i) {
        in[i] = std::complex<float>(i + 1, i + 1); // Real and imaginary parts
    }

    // Create the FFT plan
    fftwf_plan plan = fftwf_plan_dft_1d(N, fftw_in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the FFT
    fftwf_execute(plan);

    // Output the FFT result
    std::cout << "FFT Result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Element " << i << ": "
            << out[i][0] << " + " << out[i][1] << "i" << std::endl;
    }

    // Destroy the plan and free allocated memory
    fftwf_destroy_plan(plan);
    fftw_free(out);
    free(in);   

  
        //glutInit(&argc, argv); // ������������� GLUT
        //glutInitDisplayMode(GLUT_RGB); // ��������� ������ ����������� (RGB)
        //glutInitWindowSize(800, 600); // ��������� �������� ����
        //glutCreateWindow("������� ������ GLUT"); // �������� ����

        //glutDisplayFunc(display); // ��������� ������� �����������

        //glutMainLoop(); // ������ �������� ����� GLUT  
    return 0;
}
