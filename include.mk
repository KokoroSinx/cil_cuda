# Use with homebrew installed boost/hdf5/voro++
export CC  = clang
export CXX = clang++
export HDF_DIR = /usr/local
export CFLAGS  = -Ofast -Wall -Wno-unknown-pragmas -Wfatal-errors\
       	       	 -DWITHOUT_MPI -D__ArrayExtensions -D__NOARRAY2OPT \
		 -fomit-frame-pointer -fstrict-aliasing -Wstrict-aliasing=2 -ffast-math \
		 -msse2 -mfpmath=sse -DHAVE_SSE2=1 -DDSFMT_MEXP=19937 \
		 -I$(HDF_DIR)/include
# export CFLAGS  =  -Wall -Wno-unknown-pragmas -Wfatal-errors\
#        	       	 -DWITHOUT_MPI -D__ArrayExtensions -D__NOARRAY2OPT \
# 		 -fomit-frame-pointer -fstrict-aliasing -Wstrict-aliasing=2 -ffast-math \
# 		 -msse2 -mfpmath=sse -DHAVE_SSE2=1 -DDSFMT_MEXP=19937 \
# 		 -I$(HDF_DIR)/include
export CFLAGSMT= -O3 -msse2 -mfpmath=sse -fno-strict-aliasing \
       		 -DHAVE_SSE2=1 -DDSFMT_MEXP=19937 -DNDEBUG
export CXXFLAGS = -std=c++11 $(CFLAGS)
export LINKS    = -L$(HDF_DIR)/lib -lhdf5_hl -lhdf5 -lm
