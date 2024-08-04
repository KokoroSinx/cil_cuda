#ifndef COMMON_HPP
#define COMMON_HPP

#include "Utils.hpp"
#include "Array.h"

using std::string;
using std::vector;
using std::map;
using Array::array1;
using Array::array2;
using Array::array3;
using boost::format;
using boost::irange;

static const int MAX_DIM              = 3;
static const int MAX_ACTIVE_NOISE_DIM = 3;

extern int DIM;
extern int ACTIVE_NOISE_DIM;

namespace NOISE{
  enum NOISE{
    PASSIVE,
    RADIAL ,
    ANGULAR,
    _SIZE_,
  };
}

namespace OPT{
  enum OPT{
    INTEGRATOR_EULER,
    INTEGRATOR_STOCHASTIC_EULER,
    INTEGRATOR_STOCHASTIC_HEUN,
    INTEGRATOR_DPD,
    CELLS_DIVIDING,
    CELLS_DYING,
    LINKLIST_ON,
    LINKLIST_STENCIL,
    LINKLIST_SORTED,
    OUTPUT_SILO,
    OUTPUT_LAMMPS,
    OUTPUT_GSD,
    OUTPUT_H5,
    OUTPUT_DUMP,
    FORCE_SOFTCORE,
    FORCE_HERTZIAN,
    FORCE_GHOSTS,
    FORCE_EA,
    FORCE_SHIFTED,
    WALLS_ON,
    DIV_CONST_FORCE,
    SWELLING_ON,
    SWELLING_INTER_ON,
    NOISE_ON,
    NOISE_ACTIVE,
    _SIZE_,
  };
}
template<> struct enum_traits<OPT::OPT>: enum_traiter<OPT::OPT, OPT::_SIZE_>{};
extern enum_set<OPT::OPT> SWITCH;

template<typename T>
inline void printArray(std::ostream &s, format &fmt, const array1<T> &A, const double &A0 = 1.0){
  for(auto &dmy : A) s << fmt % (dmy/A0);
}

template<typename T>
inline void printArray(std::ostream &s, format &fmt_i, format &fmt_d, const array2<T> &A, const double &A0 = 1.0){
  for(auto i = 0u; i < A.Nx(); i++){
    s << fmt_i % i;
    for(auto j = 0u; j < A.Ny(); j++) s << fmt_d % (A(i,j)/A0);
    if(i != A.Nx()-1) s << '\n';
  }
}

#endif
