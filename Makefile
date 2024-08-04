include include.mk

NVCC = /usr/local/cuda/bin/nvcc

OBJS = Random.o Format.o gsd.o Input.o Output.o OutputWriter.o Wall.o MD.o Container.o Colony.o CIL.o
EXE  = x
.SUFFIXES: .c .cxx .cpp .o .out

# Obtain the version and commit numbers of the source code from git and make them accessible within the program. CAVEAT: May require a make clean to update.
# https://stackoverflow.com/questions/1704907/how-can-i-get-my-c-code-to-automatically-print-out-its-git-version-hash/12368262#12368262
GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)
GITFLAG = -DVERSION=\"$(GIT_VERSION)\"
LDFLAGS+=-undefined dynamic_lookup

PROG  = cil
all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(OBJS) mt/dSFMT.o -o $@.$(EXE) $(CFLAGS) $(LINKS) $(GITFLAG)
	$(SIGN)

%.o : %.cxx
	$(CXX) -c $< $(CXXFLAGS) $(GITFLAG) -o $@

%.o : %.cpp
	$(CXX) -c $< $(CXXFLAGS) $(GITFLAG) -o $@

%.o : %.c 
	$(CC) -c $< $(CFLAGS) -o $@
	
%.o : %.cu
	$(NVCC) -c $< -o $@ $(CFLAGS) -std=c++11 $(GITFLAG)
	
clean:
	rm -f *~ *.o ${PROG}.x
