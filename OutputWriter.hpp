#ifndef OUTPUT_WRITER_HPP
#define OUTPUT_WRITER_HPP
#include <string> // required for std::string
#include <sys/types.h> // required for stat.h
#include <sys/stat.h> // no clue why required -- man pages say so

#include "gsd.h"

#ifdef WITHSILO
#include <silo.h>
#endif

#include "Input.hpp"

void initializeOutput();
void outputFrame(int id, int ts);
void finalizeOutput();
#endif
