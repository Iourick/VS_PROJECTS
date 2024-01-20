# CUDA PATH
CUDAPATH = /usr/local/cuda-12.2/

# Compiling flags
#CFLAGS = -I/home/yuryk/VS_PROJECTS_C++/Novotny_cdmt/cdmt-master/cuda-samples-master/Common/ -I/home/yuryk/hdf/HDF5-1.14.3-Linux/HDF_Group/HDF5/1.14.3/include/ -std=c++17 
CFLAGS = -I/home//yuryk/VS_PROJECTS_C++/cdmth/cuda-samples-master/Common/ -I/home/yuryk/hdf/HDF5-1.14.3-Linux/HDF_Group/HDF5/1.14.3/include/ -std=c++17 

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -L/home/yuryk/hdf/HDF5-1.14.3-Linux/HDF_Group/HDF5/1.14.3/lib/ -lcufft -lhdf5 -lcurand /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcufft.so 
# /home/yuryk/hdf/HDF5-1.14.3-Linux/HDF_Group/HDF5/1.14.3/lib/libhdf5.so

# Compilers
NVCC = $(CUDAPATH)/bin/nvcc
CC   = /usr/bin/g++-10

cdmth: cdmth.o
	$(NVCC) $(CFLAGS) -o cdmth cdmth.o $(LFLAGS)

cdmth.o: cdmth.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o
	rm -f *~
	rm -f cdmt

