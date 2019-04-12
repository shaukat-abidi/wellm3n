#-----------------------------------------------------------------------------
# Libraries and includs
#-----------------------------------------------------------------------------
PLAT=LINUX
BLLIB = /home/sabidi/Downloads/BLAS/libblas.a
CBLIB = /home/sabidi/Shaukat/LinearAlgebra/CBLAS/lib/cblas_$(PLAT).a

#-----------------------------------------------------------------------------
# Compilers
#-----------------------------------------------------------------------------

CC = gcc
FC = gfortran
LOADER = $(FC)

#-----------------------------------------------------------------------------
# Flags for Compilers
#-----------------------------------------------------------------------------

CFLAGS = -O3 -DADD_ 
FFLAGS = -O3  

all: main_file 

main_file: mvl_common.o
	$(CC) -c $(CFLAGS) -I../../src main_file.c
	$(LOADER) -o main_file main_file.o mvl_common.o $(CBLIB) $(BLLIB)

mvl_common.o: mvl_common.c mvl_common.h 
	$(CC) -c $(CFLAGS) mvl_common.c -o mvl_common.o
   
clean:
	rm -f *.o main_file
