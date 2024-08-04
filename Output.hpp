#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "Input.hpp"
#include "Format.hpp"
#include <string> // required for std::string
#include <sys/types.h> // required for stat.h
#include <sys/stat.h> // no clue why required -- man pages say so
// Output Simulation Parameters
std::ostream & operator << (std::ostream &s, const Units &u);
std::ostream & operator << (std::ostream &s, const MDParams &p);
std::ostream & operator << (std::ostream &s, const LJParams &p);
std::ostream & operator << (std::ostream &s, const WallParams &p);
std::ostream& operator << (std::ostream& s, const CellParams &p);
std::ostream& operator << (std::ostream& s, const CellColony &c);
std::ostream& operator << (std::ostream& s, const Container &b);
std::ostream& operator << (std::ostream& s, const OutputParams &o);
#endif
