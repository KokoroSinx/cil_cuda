#ifndef RANDOM_HPP
#define RANDOM_HPP
#include <cmath>
#include <random>
#include "mt/dSFMT.h"

namespace Random{
  extern dsfmt_t  dsfmt;
  inline void init(int seed){
    dsfmt_init_gen_rand(&dsfmt, seed);
  }
  inline double uniform(){ // uniform random in (-1, 1)
    return (dsfmt_genrand_open_open(&dsfmt) - 0.5)*2.0;
  }
  inline double uniform0x(const double &x){ // uniform random in [0, x)
    return dsfmt_genrand_close_open(&dsfmt)*x;
  }
  /*!
    A fast normal random number generator
    Joseph L. Leva, ACM Transaction on Mathematical Software 18, 4, 449-453 (1992)
    http://saluc.engr.uconn.edu/refs/crypto/rng/leva92afast.pdf
    http://www.icsi.berkeley.edu/ftp/pub/speech/software/praatlib-0.3/src/GSL/gsl_randist__gauss.c
   */
  inline double normal(){
    const double s  =  0.449871;
    const double t  = -0.386595;
    const double a  =  0.19600;
    const double b  =  0.25472;
    const double r1 =  0.27597;
    const double r2 =  0.27846;
    double u, v, x, y, Q;
    do{

      // 1)
      u = 1.0 - dsfmt_genrand_close_open(&dsfmt);          // 1 - [0, 1) = (0, 1]
      v = 1.7156*(dsfmt_genrand_close_open(&dsfmt) - 0.5); // [-r, r), r = 0.8578

      // 2)
      x = u - s;
      y = fabs(v) - t;
      Q = x*x + y*(a*y - b*x);

      /*
	3) Accept point if Q is less than inner boundary         : Q < r1
	4) Reject point if Q is greater than the outer boundary  : Q > r2
	5) Reject point if outside of acceptance region          : v^2 > -4 u^2 ln(u)
       */
    }while( Q >= r1 &&
	    (Q > r2 || v*v > -4.0*u*u*log(u)));

    return (v/u);
  }
  inline double normal(const double &mu, const double &sigma){return mu + sigma*normal();}

	inline double logNormal(const double &mu, const double &sigma){
		auto Z = Random::normal(0, 1);
		auto mu2 = log((mu*mu)/(sqrt(mu*mu+sigma*sigma)));
		auto sigma2 = sqrt(log(1+(sigma*sigma)/(mu*mu)));
		return exp(mu2+sigma2*Z);
	}

  inline double insideCircle(double &a, double &b){
    int inside = 0;
    double r2  = 0.0;
    do{
      a = uniform();
      b = uniform();
      r2= a*a + b*b;
      if(r2 <= 1.0) inside = 1;
    }while(!inside);
    return r2;
  }
  inline double insideSphere(double &a, double &b, double &c){
    int inside = 0;
    double r2  = 0.0;
    do{
      a = uniform();
      b = uniform();
      c = uniform();
      r2= a*a + b*b + c*c;
      if(r2 <= 1.0) inside = 1;
    }while(!inside);
    return r2;
  }
  inline double insideDSphere(const int &dim, double *ni){
    assert(dim == 2 || dim == 3);
    int inside = 0;
    double r2  = 0.0;
    do{
      r2 = 0.0;
      for(int d = 0; d < dim; d++){
         ni[d] = uniform();
         r2 += ni[d]*ni[d];
       }
      if(r2 <= 1.0) inside = 1;
    }while(!inside);
    return r2;
  }

  inline void onCircle(double &a, double &b){
    double norm = 1.0 / sqrt(insideCircle(a, b));
    a *= norm;
    b *= norm;
  }
  inline void onSphere(double &a, double &b, double &c){
    double norm = 1.0 / sqrt(insideSphere(a, b, c));
    a *= norm;
    b *= norm;
    c *= norm;
  }
  inline void onDSphere(const int &dim, double *ni){
    double norm = 1.0 / sqrt(insideDSphere(dim, ni));
    for(int d = 0; d < dim; d++) ni[d] *= norm;
  }
}

#endif
