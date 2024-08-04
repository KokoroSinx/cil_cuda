#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <limits>
#include <bitset>
#include <stdexcept>
#include <cassert>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/format.hpp>
#include <boost/range/irange.hpp>

// openmp support
#ifdef _OPENMP
#include <omp.h>
#else
#ifndef __clang__

#include <chrono>

#else
#include <ctime>
#endif
#endif

#ifdef USE_RESTRICT
#else
#define __restrict
#endif

namespace Constants {
    static const double PI = M_PI;
    static const double TWOPI = 2.0 * M_PI;
    static const double PI_HALF = PI / 2.0;
    static const double SIXTH_ROOT_TWO = pow(2.0, 1.0 / 6.0);
    static const double THIRD_ROOT_TWO = SIXTH_ROOT_TWO * SIXTH_ROOT_TWO;
    static const double EPSILON_MP = std::numeric_limits<double>::epsilon(); //1.0e-15
    static const double MAX_MP = std::numeric_limits<double>::max();
    static const double TOL_MP = 10.0 * EPSILON_MP;   //1.0e-14
    static const double BIG_TOL_MP = 1.0e3 * TOL_MP;      //1.0e-11
    static const double BIGG_TOL_MP = 1.0e3 * BIG_TOL_MP;  //1.0e-8
    static const double HUGE_TOL_MP = 1.0e3 * BIGG_TOL_MP; //1.0e-5
    static const double KB = 1.38064852e-23;            //Boltzmann constant with thi unit of J/K^(-1)
}

inline void init_threads() {
    using std::cerr;
    using std::endl;
#ifdef _OPENMP
                                                                                                                            int nthreads, tid, procs, maxt, inpar, dynamic, nested;
  #pragma omp parallel private(nthreads, tid)
  {
    tid = omp_get_thread_num();
    if(tid == 0){
      procs   = omp_get_thread_num();
      nthreads= omp_get_num_threads();
      maxt    = omp_get_max_threads();
      dynamic   =  omp_get_max_active_levels();
      inpar  = omp_in_parallel();
      nested  = omp_get_nested();

      cerr << "# " << endl;
      cerr << "# OMP RUNTIME :" << endl;
      cerr << "# Number of processors        = " << procs    << endl;
      cerr << "# Number of threads           = " << nthreads << endl;
      cerr << "# Max threads                 = " << maxt     << endl;
      cerr << "# In parallel ?               = " << inpar    << endl;
      cerr << "# Dynamic thread enabled ?    = " << dynamic  << endl;
      cerr << "# Nested parallelism enabled? = " << nested   << endl;
      cerr << "# " << endl;
    }
  }
#endif
}


inline std::string tolower(const std::string &dmy) {
    std::string low(dmy);
    for (auto i : boost::irange(0ul, low.size())) low[i] = tolower(low[i]);
    return low;
}

inline std::string toupper(const std::string &dmy) {
    std::string upp(dmy);
    for (auto i : boost::irange(0ul, upp.size())) upp[i] = toupper(upp[i]);
    return upp;
}

inline void run_assert(const bool &p) {
    if (!p) throw std::runtime_error("runtime assert failed!");
}

inline void run_assert(const bool &p, const std::string &arg) {
    if (!p) throw std::runtime_error(arg);
}

inline void run_assert(const bool &p, const char *arg) {
    if (!p) throw std::runtime_error(arg);
}

inline unsigned int lshift(const unsigned int &i, const unsigned int &imin, const unsigned int &imax) {
    return (i + 1 < imax ? i + 1 : imin);
}

inline unsigned int rshift(const unsigned int &i, const unsigned int &imin, const unsigned int &imax) {
    return (i > imin ? i - 1 : imax - 1);
}

template<typename T>
inline bool non_zero_mp(const T &a) {
    return (a > Constants::TOL_MP) || (-Constants::TOL_MP > a);
}

template<typename T>
inline bool zero_mp(const T &a) {
    return (a <= Constants::TOL_MP && (-Constants::TOL_MP <= a));
}

template<typename T>
inline bool positive_mp(const T &a) {
    return a > Constants::TOL_MP;
}

template<typename T>
inline bool negative_mp(const T &a) {
    return -Constants::TOL_MP > a;
}

/*
e_1 = (y, -x,0) -> perpendicular (~ x axis)
e_2 = (x, y, 0) -> parallel      (~ y axis)
| i  j  k |
| y -x  0 |  = e_1 \cross e_2 = e_3 = k (y^2 + x^2) > 0
| x  y  0 |
*/
class Projector2D {
private:
    double x, y; // components of e_2 (parallel vector)
    inline void normalize() {
        double norm = 1.0 / sqrt(x * x + y * y);
        x *= norm;
        y *= norm;
    }

public:
    Projector2D(const double &_x, const double &_y) : x(_x), y(_y) { this->normalize(); }

    Projector2D(const double r[2]) : Projector2D(r[0], r[1]) {}

    //return parallel component : dot(r, e_2)
    double P(const double r[2]) const {
        return r[0] * x + r[1] * y;
    }

    //return perpendicular component : dot(r, e_1)
    double Q(const double r[2]) const {
        return r[0] * y - r[1] * x;
    }
};

// nearest integer
template<typename T>
inline int Nint(const T &a) {
    return (a >= 0.0 ? static_cast<int>(a + 0.5) : static_cast<int>(a - 0.5));
}

// square
template<typename T>
inline __host__ __device__ T SQ(const T &a) {
    return a * a;
}

template<typename T>
inline T POW3(const T &a) {
    return a * a * a;
}

// minimum
template<typename T>
inline T MIN(const T &a, const T &b) {
    return (a <= b ? a : b);
}

// maximum
template<typename T>
inline T MAX(const T &a, const T &b) {
    return (a >= b ? a : b);
}

// absolute value
template<typename T>
inline T ABS(const T &a) {
    return (a >= 0 ? a : -a);
}

template<typename T>
inline void SWAP(T &a, T &b) {
    T a_cp = a;
    a = b;
    b = a;
}

inline double dot(const double x1, const double y1, const double x2, const double y2){
    return x1*x2+y1*y2;
}

inline double length(const double x, const double y){
    return sqrt(x*x+y*y);
}



inline double Sigma2Volume(const double &a) {
    return 4 * Constants::PI * POW3(a / 2) / 3;
}

inline double Volume2Sigma(const double &a) {
    return 2 * cbrt(3 * a / (4 * Constants::PI));
    // cbrt(a/4*Constants::PI) (4*Constants::PI*POW3(a))/3
}

inline double Sigma2Area(const double &a) {
    return Constants::PI * SQ(a / 2);
}

inline double twoSigma2ContactArea(const double &r, const double &a, const double &b) {
    auto ra = a / 2;
    auto rb = b / 2;
    auto tempA = sqrt(4 * SQ(r) * SQ(ra) - SQ(SQ(r) - SQ(rb) + SQ(ra))) / (2 * r);
    return (tempA > 0 ? Constants::PI * SQ(tempA) : 0);
}

inline double twoSigma2ContactLength(const double &a, const double &b, const double &h) {
	auto ra = a / 2;
	auto rb = b / 2;
	auto tempA = 2*sqrt(SQ(ra)-SQ(ra-(2*rb*h-SQ(h))/(2*(ra+rb-h))));
	auto small_r = (a<=b) ? a:b;
	tempA = isnan(tempA)? small_r:tempA;
	return tempA;
}

inline double Area2Sigma(const double &a) {
    return 2 * sqrt(a / Constants::PI);
    // cbrt(a/4*Constants::PI) (4*Constants::PI*POW3(a))/3
}

inline double equal_tol(const double &a, const double &b, const double &rtol, const double &atol = Constants::TOL_MP) {
    if (a == b) return true;

    double diff = ABS(a - b);
    if (diff <= atol) return true;

    double eps = MAX(ABS(a), ABS(b)) * rtol;
    return (diff <= eps ? true : false);
}

inline bool equal_mp(const double &a, const double &b) {
    if (a == b) return true;
    double eps = (MAX(ABS(a), ABS(b)) + 10.0) * Constants::EPSILON_MP;
    return (ABS(a - b) <= eps ? true : false);
}

inline bool dirCheckMake(const char *dname) {
    DIR *dtest;
    if ((dtest = opendir(dname)) == NULL) {
        char dmy_cmd[256];
        snprintf(dmy_cmd, 256, "mkdir %s", dname);
        system(dmy_cmd);
        DIR *dnew;
        if ((dnew = opendir(dname)) == NULL) {
            fprintf(stderr, "Error: failed to create directory\n");
            return false;
        }
        closedir(dnew);
    } else {
        closedir(dtest);
    }
    return true;
}

inline bool fileCheck(const char *fname) {
    FILE *ftest;
    if (NULL == (ftest = fopen(fname, "r"))) {
        return false;
    }
    fclose(ftest);
    return true;
}

// http://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of
template<typename T, typename Compare>
std::vector<int> sort_permutation(std::vector < T >
const&vec,
Compare compare
){
std::vector<int> p(vec.size());
std::iota(p
.

begin(), p

.

end(),

0);
std::sort(p
.

begin(), p

.

end(),

[&](
int i,
int j
){
return
compare(vec[i], vec[j]
);});
return
p;
}
template<typename T>
std::vector <T> apply_permutation(std::vector < T >
const& vec,
std::vector<int> const &p
){
std::vector <T> sorted_vec(p.size());
std::transform(p
.

begin(), p

.

end(), sorted_vec

.

begin(),

[&](
int i
){
return vec[i]; });
return
sorted_vec;
}

struct WallTimer {
#ifndef _OPENMP
#ifndef __clang__
    std::chrono::time_point <std::chrono::system_clock> t_start;

    inline void start() { t_start = std::chrono::system_clock::now(); }

    inline double stop() {
        std::chrono::time_point <std::chrono::system_clock> t_end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;
        return elapsed.count();
    }

#else
                                                                                                                            time_t t_start;
  inline void start(){time(&t_start);}
  inline double stop(){
    time_t t_end;
    time(&t_end);
    return static_cast<double>(difftime(t_end, t_start));
  }
#endif
#else
                                                                                                                            double t_start;
  inline void start(){t_start = omp_get_wtime();}
  inline double stop(){return (omp_get_wtime() - t_start);}
#endif
};

// MD utilities for orthorombic unit cell with origin at 0
namespace MDutils {
    using boost::irange;

    // compute pbc distance
    template<typename Q, typename R, typename S>
    inline double distance(double *r12,
                           const Q &r1, const R &r2,
                           const S &lbox, const int &dim) {
        double dmy_dist = 0.0;
        for (auto d : irange(0, dim)) {
            r12[d] = r2[d] - r1[d];
            r12[d] -= static_cast<double>(Nint(r12[d] / lbox[d])) * lbox[d];
            dmy_dist += SQ(r12[d]);
        }
        return dmy_dist;
    }

    template<typename Q, typename R, typename S, typename T>
    inline double distance(double *r12,
                           const Q &r1, const R &r2,
                           const S &lbox, const T &pbc_boundary, const int &dim) {
        assert(dim > 0);
        double dmy_dist = 0.0;
        for (auto d : irange(0, dim)) {
            r12[d] = r2[d] - r1[d];
            if (pbc_boundary[d]) r12[d] -= static_cast<double>(Nint(r12[d] / lbox[d])) * lbox[d];
            dmy_dist += SQ(r12[d]);
        }
        return dmy_dist;
    }

    template<typename Q, typename R, typename S>
    inline double distance2D(double *r12, const Q &r1, const R &r2, const S &lbox) {
        r12[0] = r2[0] - r1[0];
        r12[0] -= static_cast<double>(Nint(r12[0] / lbox[0])) * lbox[0];

        r12[1] = r2[1] - r1[1];
        r12[1] -= static_cast<double>(Nint(r12[1] / lbox[1])) * lbox[1];

        return (SQ(r12[0]) + SQ(r12[1]));
    }

    template<typename Q, typename R, typename S, typename T>
    inline double distance2D(double *r12, const Q &r1, const R &r2, const S &lbox,
                             const T &pbc_boundary) {
        r12[0] = r2[0] - r1[0];
        r12[1] = r2[1] - r1[1];

        if (pbc_boundary[0]) r12[0] -= static_cast<double>(Nint(r12[0] / lbox[0])) * lbox[0];
        if (pbc_boundary[1]) r12[1] -= static_cast<double>(Nint(r12[1] / lbox[1])) * lbox[1];

        return (SQ(r12[0]) + SQ(r12[1]));
    }

    class BoundingBox {
    private:
        static const int MAX_DIM = 3;
        int DIM;
        double lengths[MAX_DIM];
        bool pbcs[MAX_DIM];
    public:
        BoundingBox() : DIM(MAX_DIM), lengths{0.0, 0.0, 0.0}, pbcs{false, false, false} {}

        template<typename S, typename T>
        void init(const int &dim, const S &lbox, const T &pbc_boundary) {
            run_assert(dim > 0 && dim <= MAX_DIM);
            DIM = dim;
            for (int d = 0; d < DIM; d++) {
                run_assert(lbox[d] > 0.0, "box dimensions should be positive");
                lengths[d] = lbox[d];
            }
            for (int d = 0; d < DIM; d++) pbcs[d] = pbc_boundary[d];
        }

        template<typename T>
        inline void resetBox(const T &lbox) {
            for (int d = 0; d < DIM; d++) {
                run_assert(lbox[d] > 0.0, "box dimensions should be positive");
                lengths[d] = lbox[d];
            }
        }

        inline int dim() const { return DIM; }

        inline void length(std::vector<double> &ls) const {
            ls.resize(DIM);
            for (int d = 0; d < DIM; d++) ls[d] = lengths[d];
        }

        inline void length(double *ls) const {
            for (int d = 0; d < DIM; d++) ls[d] = lengths[d];
        }

        inline double lx() const { return lengths[0]; }

        inline double ly() const { return lengths[1]; }

        inline double lz() const { return (DIM == 3 ? lengths[2] : 0.0); }

        inline double maxSize() const {
            double xymax = MAX(lengths[0], lengths[1]);
            return (DIM == 2 ? xymax : MAX(xymax, lengths[2]));
        }

        inline double minSize() const {
            double xymin = MIN(lengths[0], lengths[1]);
            return (DIM == 2 ? xymin : MIN(xymin, lengths[2]));
        }

        inline void isPeriodic(std::vector<bool> &pb) const {
            pb.resize(DIM);
            for (int d = 0; d < DIM; d++) pb[d] = pbcs[d];
        }

        inline void isPeriodic(bool *pb) const {
            for (int d = 0; d < DIM; d++) pb[d] = pbcs[d];
        }

        inline bool isPeriodicX() const { return pbcs[0]; }

        inline bool isPeriodicY() const { return pbcs[1]; }

        inline bool isPeriodicZ() const { return (DIM == 3 ? pbcs[2] : false); }

        inline bool isValidPos(double *const r) const {
            bool ans = true;
            for (int d = 0; d < DIM; d++) ans = ans && (!pbcs[d] || (r[d] >= 0.0 && r[d] < lengths[d]));
            return ans;
        }

        inline bool isValidFrc(double *const f) const {
        		bool ans = true;
						for (int d = 0; d < DIM; d++) ans = ans && !isnan(f[d]);
						return ans;
        }

        inline bool isBoundedPos(double *const r) const {
            bool ans = true;
            for (int d = 0; d < DIM; d++) ans = ans && (r[d] >= 0.0 && r[d] < lengths[d]);
            return ans;
        }

        template<typename S, typename T>
        inline __host__ __device__ double distance(double *const r12, const S &r1, const T &r2) const {
            double ans = 0.0;
            for (int d = 0; d < DIM; d++) {
                r12[d] = r2[d] - r1[d];
                if (pbcs[d]) r12[d] -= static_cast<double>(Nint(r12[d] / lengths[d])) * lengths[d];
                ans += SQ(r12[d]);
            }
            return ans;
        }

        inline void pbc(double *const r) const {
            for (int d = 0; d < DIM; d++) r[d] = (pbcs[d] ? fmod(r[d] + lengths[d], lengths[d]) : r[d]);
            assert(isValidPos(r));
        }

        inline double updateTor(double orien, double const torque, const double dt) const {
            auto ans = fmod(orien += torque*dt, 3.1415);
            return ans;
        }

        inline void updatePos(double *const r, double const *const dr) const {
            for (int d = 0; d < DIM; d++) r[d] = (pbcs[d] ? fmod(r[d] + dr[d] + lengths[d], lengths[d]) : r[d] + dr[d]);
            assert(isValidPos(r));
        }

        inline void updatePos(double *const r, double const *const v, const double dt) const {
            for (int d = 0; d < DIM; d++) {
                r[d] = (pbcs[d] ? fmod(r[d] + dt * v[d] + lengths[d], lengths[d]) : r[d] + v[d] * dt);
            }
            assert(isValidPos(r));
        }

        inline void updateVel(double *const v, double const *const fic, double const *const fid, double const *const fir,
                  const double dt, const double mass) const {
            for (int d = 0; d < DIM; d++) v[d] = v[d] + (fic[d] * dt + fid[d] * dt + fir[d] * sqrt(dt)) / (2 * mass);
        }

        inline void updateVel(double *const v, double *const r0, double *const r1, const double dt) const {
            for (int d = 0; d < DIM; d++) v[d] = (r1[d] - r0[d]) / dt;
        }

			inline void
			updateVel(double *const v, double const *const fic, double const *const fid, const double tau_ve_dt, const double zeta) const {
				for (int d = 0; d < DIM; d++) v[d] = ((1+(tau_ve_dt))/zeta)*(fic[d])+(tau_ve_dt/zeta)*fid[d];
			}

        inline void updateVel(double *const v, double const *const fic, const double zeta) const {
            for (int d = 0; d < DIM; d++) v[d] = fic[d]*zeta;
        }
        inline void updateVel(double *const v, double const r00, double const r01, double const r10, double const r11,
                              const double dt) const {
            v[0] = (r10 - r00) / dt;
            v[1] = (r11 - r01) / dt;
        }

        inline void resetVel(double *const v) const {
            for (int d = 0; d < DIM; d++) v[d] = 0;
        }

        inline void resetFrc(double *const fic, double *const fid, double *const fir) const {
            for (int d = 0; d < DIM; d++) {
                fic[d] = 0;
                fid[d] = 0;
                fir[d] = 0;
            }
        }
    };

    template<typename T>
    inline double distance3D(double *r12, const T &lbox) {
        r12[0] -= static_cast<double>(Nint(r12[0] / lbox[0])) * lbox[0];
        r12[1] -= static_cast<double>(Nint(r12[1] / lbox[1])) * lbox[1];
        r12[2] -= static_cast<double>(Nint(r12[2] / lbox[2])) * lbox[2];
        return (SQ(r12[0]) + SQ(r12[1]) + SQ(r12[2]));
    }

    template<typename Q, typename R, typename S>
    inline double distance3D(double *r12, const Q &r1, const R &r2, const S &lbox) {
        r12[0] = r2[0] - r1[0];
        r12[0] -= static_cast<double>(Nint(r12[0] / lbox[0])) * lbox[0];

        r12[1] = r2[1] - r1[1];
        r12[1] -= static_cast<double>(Nint(r12[1] / lbox[1])) * lbox[1];

        r12[2] = r2[2] - r1[2];
        r12[2] -= static_cast<double>(Nint(r12[2] / lbox[2])) * lbox[2];

        return (SQ(r12[0]) + SQ(r12[1]) + SQ(r12[2]));
    }

    template<typename Q, typename R, typename S, typename T>
    inline double distance3D(double *r12, const Q &r1, const R &r2, const S &lbox,
                             const T &pbc_boundary) {
        r12[0] = r2[0] - r1[0];
        r12[1] = r2[1] - r1[1];
        r12[2] = r2[2] - r1[2];

        if (pbc_boundary[0]) r12[0] -= static_cast<double>(Nint(r12[0] / lbox[0])) * lbox[0];
        if (pbc_boundary[1]) r12[1] -= static_cast<double>(Nint(r12[1] / lbox[1])) * lbox[1];
        if (pbc_boundary[2]) r12[2] -= static_cast<double>(Nint(r12[2] / lbox[2])) * lbox[2];
        return (SQ(r12[0])) + SQ(r12[1]) + SQ(r12[2]);
    }

    // place particle inside box: r_i in [0,L_i]
    template<typename R>
    inline void pbc(double *x, const R &lbox, const int &dim) {
        for (auto d : irange(0, dim)) {
            x[d] = fmod(x[d] + lbox[d], lbox[d]);
            assert(x[d] >= 0.0 && x[d] < lbox[d]);
        }
    }

    template<typename R, typename S>
    inline void pbc(double *x, const R &lbox, const S &pbc_boundary, const int &dim) {
        for (auto d : irange(0, dim)) {
            x[d] = (pbc_boundary[d] ? fmod(x[d] + lbox[d], lbox[d]) : x[d]);
            assert(!pbc_boundary[d] || (x[d] >= 0.0 && x[d] < lbox[d]));
        }
    }

    template<typename R>
    inline void pbc2D(double *x, const R &lbox) {
        x[0] = fmod(x[0] + lbox[0], lbox[0]);
        x[1] = fmod(x[1] + lbox[1], lbox[1]);

        assert(x[0] >= 0.0 && x[0] < lbox[0]);
        assert(x[1] >= 0.0 && x[1] < lbox[1]);
    }

    template<typename R, typename S>
    inline void pbc2D(double *x, const R &lbox, const S &pbc_boundary) {
        x[0] = (pbc_boundary[0] ? fmod(x[0] + lbox[0], lbox[0]) : x[0]);
        x[1] = (pbc_boundary[1] ? fmod(x[1] + lbox[1], lbox[1]) : x[1]);

        assert(!pbc_boundary[0] || (x[0] >= 0.0 && x[0] < lbox[0]));
        assert(!pbc_boundary[1] || (x[1] >= 0.0 && x[1] < lbox[1]));
    }

    template<typename R>
    inline void pbc3D(double *x, const R &lbox) {
        x[0] = fmod(x[0] + lbox[0], lbox[0]);
        x[1] = fmod(x[1] + lbox[1], lbox[1]);
        x[2] = fmod(x[2] + lbox[2], lbox[2]);

        assert(x[0] >= 0.0 && x[0] < lbox[0]);
        assert(x[1] >= 0.0 && x[1] < lbox[1]);
        assert(x[2] >= 0.0 && x[2] < lbox[2]);
    }

    template<typename R, typename S>
    inline void pbc3D(double *x, const R &lbox, const S &pbc_boundary) {
        x[0] = (pbc_boundary[0] ? fmod(x[0] + lbox[0], lbox[0]) : x[0]);
        x[1] = (pbc_boundary[1] ? fmod(x[1] + lbox[1], lbox[1]) : x[1]);
        x[2] = (pbc_boundary[2] ? fmod(x[2] + lbox[2], lbox[2]) : x[2]);

        assert(!pbc_boundary[0] || (x[0] >= 0.0 && x[0] < lbox[0]));
        assert(!pbc_boundary[1] || (x[1] >= 0.0 && x[1] < lbox[1]));
        assert(!pbc_boundary[2] || (x[2] >= 0.0 && x[2] < lbox[2]));
    }

    class LinkListOrtho {
    protected:
        static const int MAX_DIM = 3;
        static const int MAX_NEIGHBORS = 14;
        std::vector<int> DNS2D = {0, 0, 1, 0, 1, 1, 0, 1, -1, 1}; // 2 * 5
        std::vector<int> DNS3D = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                  -1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
                                  0, 1, 1, -1, 1, 1, -1, 0, 1, -1, -1, 1,
                                  0, -1, 1, 1, -1, 1}; // 3 * 14

        std::vector<int> DNS3DBIG = {-1, -1, 1, -1, 0, 1, -1, 1, 0, -1, 1, 1, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                                     1,
                                     1, -1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, -5, -3, 1, -5, -2, 1, -5, -2, 2, -5,
                                     -1,
                                     1, -5, -1, 2, -5, -1, 3, -5, 0, 1, -5, 0, 2, -5, 0, 3, -5, 1, 0, -5, 1, 1, -5, 1,
                                     2,
                                     -5, 1, 3, -5, 2, 0, -5, 2, 1, -5, 2, 2, -5, 3, 0, -5, 3, 1, -4, -4, 1, -4, -3, 1,
                                     -4,
                                     -3, 2, -4, -3, 3, -4, -2, 1, -4, -2, 2, -4, -2, 3, -4, -1, 1, -4, -1, 2, -4, -1, 3,
                                     -4, -1, 4, -4, 0, 1, -4, 0, 2, -4, 0, 3, -4, 0, 4, -4, 1, 0, -4, 1, 1, -4, 1, 2,
                                     -4,
                                     1, 3, -4, 1, 4, -4, 2, 0, -4, 2, 1, -4, 2, 2, -4, 2, 3, -4, 3, 0, -4, 3, 1, -4, 3,
                                     2,
                                     -4, 3, 3, -4, 4, 0, -4, 4, 1, -3, -5, 1, -3, -4, 1, -3, -4, 2, -3, -4, 3, -3, -3,
                                     1,
                                     -3, -3, 2, -3, -3, 3, -3, -3, 4, -3, -2, 1, -3, -2, 2, -3, -2, 3, -3, -2, 4, -3,
                                     -1,
                                     1, -3, -1, 2, -3, -1, 3, -3, -1, 4, -3, -1, 5, -3, 0, 1, -3, 0, 2, -3, 0, 3, -3, 0,
                                     4,
                                     -3, 0, 5, -3, 1, 0, -3, 1, 1, -3, 1, 2, -3, 1, 3, -3, 1, 4, -3, 1, 5, -3, 2, 0, -3,
                                     2,
                                     1, -3, 2, 2, -3, 2, 3, -3, 2, 4, -3, 3, 0, -3, 3, 1, -3, 3, 2, -3, 3, 3, -3, 3, 4,
                                     -3,
                                     4, 0, -3, 4, 1, -3, 4, 2, -3, 4, 3, -3, 5, 0, -3, 5, 1, -2, -5, 1, -2, -5, 2, -2,
                                     -4,
                                     1, -2, -4, 2, -2, -4, 3, -2, -3, 1, -2, -3, 2, -2, -3, 3, -2, -3, 4, -2, -2, 1, -2,
                                     -2, 2, -2, -2, 3, -2, -2, 4, -2, -2, 5, -2, -1, 1, -2, -1, 2, -2, -1, 3, -2, -1, 4,
                                     -2, -1, 5, -2, 0, 1, -2, 0, 2, -2, 0, 3, -2, 0, 4, -2, 0, 5, -2, 1, 0, -2, 1, 1,
                                     -2,
                                     1, 2, -2, 1, 3, -2, 1, 4, -2, 1, 5, -2, 2, 0, -2, 2, 1, -2, 2, 2, -2, 2, 3, -2, 2,
                                     4,
                                     -2, 2, 5, -2, 3, 0, -2, 3, 1, -2, 3, 2, -2, 3, 3, -2, 3, 4, -2, 4, 0, -2, 4, 1, -2,
                                     4,
                                     2, -2, 4, 3, -2, 5, 0, -2, 5, 1, -2, 5, 2, -1, -5, 1, -1, -5, 2, -1, -5, 3, -1, -4,
                                     1,
                                     -1, -4, 2, -1, -4, 3, -1, -4, 4, -1, -3, 1, -1, -3, 2, -1, -3, 3, -1, -3, 4, -1,
                                     -3,
                                     5, -1, -2, 1, -1, -2, 2, -1, -2, 3, -1, -2, 4, -1, -2, 5, -1, -1, 2, -1, -1, 3, -1,
                                     -1, 4, -1, -1, 5, -1, 0, 2, -1, 0, 3, -1, 0, 4, -1, 0, 5, -1, 1, 2, -1, 1, 3, -1,
                                     1,
                                     4, -1, 1, 5, -1, 2, 0, -1, 2, 1, -1, 2, 2, -1, 2, 3, -1, 2, 4, -1, 2, 5, -1, 3, 0,
                                     -1,
                                     3, 1, -1, 3, 2, -1, 3, 3, -1, 3, 4, -1, 3, 5, -1, 4, 0, -1, 4, 1, -1, 4, 2, -1, 4,
                                     3,
                                     -1, 4, 4, -1, 5, 0, -1, 5, 1, -1, 5, 2, -1, 5, 3, 0, -5, 1, 0, -5, 2, 0, -5, 3, 0,
                                     -4,
                                     1, 0, -4, 2, 0, -4, 3, 0, -4, 4, 0, -3, 1, 0, -3, 2, 0, -3, 3, 0, -3, 4, 0, -3, 5,
                                     0,
                                     -2, 1, 0, -2, 2, 0, -2, 3, 0, -2, 4, 0, -2, 5, 0, -1, 2, 0, -1, 3, 0, -1, 4, 0, -1,
                                     5,
                                     0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 2, 0, 0,
                                     2,
                                     1, 0, 2, 2, 0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 3, 0, 0, 3, 1, 0, 3, 2, 0, 3, 3, 0, 3, 4,
                                     0,
                                     3, 5, 0, 4, 0, 0, 4, 1, 0, 4, 2, 0, 4, 3, 0, 4, 4, 0, 5, 0, 0, 5, 1, 0, 5, 2, 0, 5,
                                     3,
                                     1, -5, 1, 1, -5, 2, 1, -5, 3, 1, -4, 1, 1, -4, 2, 1, -4, 3, 1, -4, 4, 1, -3, 1, 1,
                                     -3,
                                     2, 1, -3, 3, 1, -3, 4, 1, -3, 5, 1, -2, 1, 1, -2, 2, 1, -2, 3, 1, -2, 4, 1, -2, 5,
                                     1,
                                     -1, 2, 1, -1, 3, 1, -1, 4, 1, -1, 5, 1, 0, 2, 1, 0, 3, 1, 0, 4, 1, 0, 5, 1, 1, 2,
                                     1,
                                     1, 3, 1, 1, 4, 1, 1, 5, 1, 2, 0, 1, 2, 1, 1, 2, 2, 1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 3,
                                     0,
                                     1, 3, 1, 1, 3, 2, 1, 3, 3, 1, 3, 4, 1, 3, 5, 1, 4, 0, 1, 4, 1, 1, 4, 2, 1, 4, 3, 1,
                                     4,
                                     4, 1, 5, 0, 1, 5, 1, 1, 5, 2, 1, 5, 3, 2, -5, 1, 2, -5, 2, 2, -4, 1, 2, -4, 2, 2,
                                     -4,
                                     3, 2, -3, 1, 2, -3, 2, 2, -3, 3, 2, -3, 4, 2, -2, 1, 2, -2, 2, 2, -2, 3, 2, -2, 4,
                                     2,
                                     -2, 5, 2, -1, 1, 2, -1, 2, 2, -1, 3, 2, -1, 4, 2, -1, 5, 2, 0, 0, 2, 0, 1, 2, 0, 2,
                                     2,
                                     0, 3, 2, 0, 4, 2, 0, 5, 2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 1, 3, 2, 1, 4, 2, 1, 5, 2, 2,
                                     0,
                                     2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 2, 5, 2, 3, 0, 2, 3, 1, 2, 3, 2, 2, 3, 3, 2,
                                     3,
                                     4, 2, 4, 0, 2, 4, 1, 2, 4, 2, 2, 4, 3, 2, 5, 0, 2, 5, 1, 2, 5, 2, 3, -5, 1, 3, -4,
                                     1,
                                     3, -4, 2, 3, -4, 3, 3, -3, 1, 3, -3, 2, 3, -3, 3, 3, -3, 4, 3, -2, 1, 3, -2, 2, 3,
                                     -2,
                                     3, 3, -2, 4, 3, -1, 1, 3, -1, 2, 3, -1, 3, 3, -1, 4, 3, -1, 5, 3, 0, 0, 3, 0, 1, 3,
                                     0,
                                     2, 3, 0, 3, 3, 0, 4, 3, 0, 5, 3, 1, 0, 3, 1, 1, 3, 1, 2, 3, 1, 3, 3, 1, 4, 3, 1, 5,
                                     3,
                                     2, 0, 3, 2, 1, 3, 2, 2, 3, 2, 3, 3, 2, 4, 3, 3, 0, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3,
                                     4,
                                     3, 4, 0, 3, 4, 1, 3, 4, 2, 3, 4, 3, 3, 5, 0, 3, 5, 1, 4, -4, 1, 4, -3, 1, 4, -3, 2,
                                     4,
                                     -3, 3, 4, -2, 1, 4, -2, 2, 4, -2, 3, 4, -1, 1, 4, -1, 2, 4, -1, 3, 4, -1, 4, 4, 0,
                                     0,
                                     4, 0, 1, 4, 0, 2, 4, 0, 3, 4, 0, 4, 4, 1, 0, 4, 1, 1, 4, 1, 2, 4, 1, 3, 4, 1, 4, 4,
                                     2,
                                     0, 4, 2, 1, 4, 2, 2, 4, 2, 3, 4, 3, 0, 4, 3, 1, 4, 3, 2, 4, 3, 3, 4, 4, 0, 4, 4, 1,
                                     5,
                                     -3, 1, 5, -2, 1, 5, -2, 2, 5, -1, 1, 5, -1, 2, 5, -1, 3, 5, 0, 0, 5, 0, 1, 5, 0, 2,
                                     5,
                                     0, 3, 5, 1, 0, 5, 1, 1, 5, 1, 2, 5, 1, 3, 5, 2, 0, 5, 2, 1, 5, 2, 2, 5, 3, 0, 5, 3,
                                     1,
                                     -5, -5, 1, -5, -5, 2, -5, -5, 3, -5, -5, 4, -5, -5, 5, -5, -4, 1, -5, -4, 2, -5,
                                     -4,
                                     3, -5, -4, 4, -5, -4, 5, -5, -3, 2, -5, -3, 3, -5, -3, 4, -5, -3, 5, -5, -2, 3, -5,
                                     -2, 4, -5, -2, 5, -5, -1, 4, -5, -1, 5, -5, 0, 4, -5, 0, 5, -5, 1, 4, -5, 1, 5, -5,
                                     2,
                                     3, -5, 2, 4, -5, 2, 5, -5, 3, 2, -5, 3, 3, -5, 3, 4, -5, 3, 5, -5, 4, 0, -5, 4, 1,
                                     -5,
                                     4, 2, -5, 4, 3, -5, 4, 4, -5, 4, 5, -5, 5, 0, -5, 5, 1, -5, 5, 2, -5, 5, 3, -5, 5,
                                     4,
                                     -5, 5, 5, -4, -5, 1, -4, -5, 2, -4, -5, 3, -4, -5, 4, -4, -5, 5, -4, -4, 2, -4, -4,
                                     3,
                                     -4, -4, 4, -4, -4, 5, -4, -3, 4, -4, -3, 5, -4, -2, 4, -4, -2, 5, -4, -1, 5, -4, 0,
                                     5,
                                     -4, 1, 5, -4, 2, 4, -4, 2, 5, -4, 3, 4, -4, 3, 5, -4, 4, 2, -4, 4, 3, -4, 4, 4, -4,
                                     4,
                                     5, -4, 5, 0, -4, 5, 1, -4, 5, 2, -4, 5, 3, -4, 5, 4, -4, 5, 5, -3, -5, 2, -3, -5,
                                     3,
                                     -3, -5, 4, -3, -5, 5, -3, -4, 4, -3, -4, 5, -3, -3, 5, -3, -2, 5, -3, 2, 5, -3, 3,
                                     5,
                                     -3, 4, 4, -3, 4, 5, -3, 5, 2, -3, 5, 3, -3, 5, 4, -3, 5, 5, -2, -5, 3, -2, -5, 4,
                                     -2,
                                     -5, 5, -2, -4, 4, -2, -4, 5, -2, -3, 5, -2, 3, 5, -2, 4, 4, -2, 4, 5, -2, 5, 3, -2,
                                     5,
                                     4, -2, 5, 5, -1, -5, 4, -1, -5, 5, -1, -4, 5, -1, 4, 5, -1, 5, 4, -1, 5, 5, 0, -5,
                                     4,
                                     0, -5, 5, 0, -4, 5, 0, 4, 5, 0, 5, 4, 0, 5, 5, 1, -5, 4, 1, -5, 5, 1, -4, 5, 1, 4,
                                     5,
                                     1, 5, 4, 1, 5, 5, 2, -5, 3, 2, -5, 4, 2, -5, 5, 2, -4, 4, 2, -4, 5, 2, -3, 5, 2, 3,
                                     5,
                                     2, 4, 4, 2, 4, 5, 2, 5, 3, 2, 5, 4, 2, 5, 5, 3, -5, 2, 3, -5, 3, 3, -5, 4, 3, -5,
                                     5,
                                     3, -4, 4, 3, -4, 5, 3, -3, 5, 3, -2, 5, 3, 2, 5, 3, 3, 5, 3, 4, 4, 3, 4, 5, 3, 5,
                                     2,
                                     3, 5, 3, 3, 5, 4, 3, 5, 5, 4, -5, 1, 4, -5, 2, 4, -5, 3, 4, -5, 4, 4, -5, 5, 4, -4,
                                     2,
                                     4, -4, 3, 4, -4, 4, 4, -4, 5, 4, -3, 4, 4, -3, 5, 4, -2, 4, 4, -2, 5, 4, -1, 5, 4,
                                     0,
                                     5, 4, 1, 5, 4, 2, 4, 4, 2, 5, 4, 3, 4, 4, 3, 5, 4, 4, 2, 4, 4, 3, 4, 4, 4, 4, 4, 5,
                                     4,
                                     5, 0, 4, 5, 1, 4, 5, 2, 4, 5, 3, 4, 5, 4, 4, 5, 5, 5, -5, 1, 5, -5, 2, 5, -5, 3, 5,
                                     -5, 4, 5, -5, 5, 5, -4, 1, 5, -4, 2, 5, -4, 3, 5, -4, 4, 5, -4, 5, 5, -3, 2, 5, -3,
                                     3,
                                     5, -3, 4, 5, -3, 5, 5, -2, 3, 5, -2, 4, 5, -2, 5, 5, -1, 4, 5, -1, 5, 5, 0, 4, 5,
                                     0,
                                     5, 5, 1, 4, 5, 1, 5, 5, 2, 3, 5, 2, 4, 5, 2, 5, 5, 3, 2, 5, 3, 3, 5, 3, 4, 5, 3, 5,
                                     5,
                                     4, 0, 5, 4, 1, 5, 4, 2, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 5, 0, 5, 5, 1, 5, 5, 2, 5, 5,
                                     3,
                                     5, 5, 4, 5, 5, 5};

        int DIM;
        int NEIGHBORS;
        int *DNS;

        int Ncells;
        int Ns[MAX_DIM];
        double iH[MAX_DIM];
        bool pbc[MAX_DIM];

        virtual int get_cellID0(int *ixyz) const = 0;

    public:
        LinkListOrtho() : DIM(MAX_DIM), NEIGHBORS(MAX_NEIGHBORS), DNS(NULL),
                          Ncells(0), Ns{1, 1, 1}, iH{0.0, 0.0, 0.0}, pbc{false, false, false} {}

        /*
            \brief Get optimum number of cells : cell size ~ rcut
            */
        void get_opt_size(int *ns, const double &rcut, const double *lbox) const {
            double delta = (*std::min_element(&lbox[0], &lbox[DIM])) * 1.0e-8;
            double ir = 1.0 / (rcut + delta);
            for (auto d : irange(0, DIM)) ns[d] = static_cast<int>(lbox[d] * ir);
        }

        /*
            \brief Initialize link list for 2d or 3d simulations
            */
        int init(int &num_neighbors, const int *ns, const double *lbox, const bool *periodic) {
            this->init_params();
            num_neighbors = NEIGHBORS;
            for (auto d : irange(0, DIM)) pbc[d] = periodic[d];
            return reset(ns, lbox);
        }

        /*
            \brief reset link list parameters manually
            */
        int reset(const int *ns, const double *lbox) {
            bool overkill = false;
            for (auto d : irange(0, DIM)) {
                assert(ns[d] > 0 && lbox[d] > 0.0);
                iH[d] = 1.0 / lbox[d];
                Ns[d] = ns[d];
                overkill = overkill || (Ns[d] < 3);
            }
            if (overkill) {
                for (auto d : irange(0, DIM)) Ns[d] = 1;
                Ncells = 1;
            } else {
                Ncells = 1;
                for (auto d : irange(0, DIM)) Ncells *= Ns[d];
            }
            assert(Ncells >= 1);
            return Ncells;
        }

        /*
            \brief Reset link list, needed when particle positions are changed
            */
        inline void reset_list(int *head, int *link, const int &totalNum) {
            for (auto i : irange(0, Ncells)) head[i] = -1;
            for (auto i : irange(0, totalNum)) link[i] = -1;
        }

        /*
            \brief Get unique cell id given cell coordinates and neighbor shift
                   Returns true/false if cell is in/out of bounds
            */
        inline int get_cellID(const int *ixyz, const int *dxyz) const {
            bool inBounds = true;
            int ls[MAX_DIM] = {0, 0, 0};

            for (auto d : irange(0, DIM)) {
                ls[d] = (ixyz[d] + dxyz[d]);
                if (pbc[d]) {
                    ls[d] = (ls[d] + Ns[d]) % Ns[d];
                    assert(ls[d] >= 0 && ls[d] < Ns[d]);
                } else if (ls[d] < 0 || ls[d] >= Ns[d]) {
                    inBounds = false;
                }
            }
            return (inBounds ? this->get_cellID0(ls) : -1);
        }

        /*
              \brief Get number of cells along each dimension
            */
        inline void get_ns(std::vector<int> &ns) {
            ns.resize(DIM);
            for (auto d : irange(0, DIM)) ns[d] = Ns[d];
        }

        /*
              \brief Get list of cell coordinates and neighbor cell shifts
            */
        inline void get_cells(int *cell, int *neighbors) const {
            int ii;
            if (cell != NULL) {
                //set cell ids
                for (auto i : irange(0, Ncells)) {
                    ii = i * DIM;
                    this->get_cellCoord(&cell[ii], i);
                }
            }
            if (neighbors != NULL) {
                for (auto i : irange(0, NEIGHBORS)) {
                    ii = i * DIM;
                    for (auto d : irange(0, DIM)) neighbors[ii + d] = DNS[ii + d];
                }
            }
        }

        /*
            Get unique cell id given cell coordinates
            TODO : consider non periodic systems
            */
        /*inline int get_cellID(const int *ixyz)const{
            int ls[DIM] = {0, 0};
            ls[0] = (ixyz[0] + Ns[0]) % Ns[0];
            ls[1] = (ixyz[1] + Ns[1]) % Ns[1];
            assert(ls[0] >= 0 && ls[0] < Ns[0]);
            assert(ls[1] >= 0 && ls[1] < Ns[1]);
            return (Ns[1]*ls[0] + ls[1]);
          }*/

        /*
          Populate link list for given cell positions
          */
        void populate_list(double *r, int *head, int *link, const int &num, const int &id0) const {

            int ls[MAX_DIM] = {0, 0, 0};
            int li, ii, k;

            for (auto i : irange(0, num)) {

                ii = DIM * i;
                for (auto d : irange(0, DIM)) ls[d] = MIN(static_cast<int>((r[ii + d] * iH[d]) * Ns[d]), Ns[d] - 1);
                k = this->get_cellID0(ls);

                li = i + id0;
                if (head[k] == -1) {
                    head[k] = li;
                } else {
                    link[li] = head[k];
                    head[k] = li;
                }
            }//i
        }

        /*
            \brief  Initialize dimensional parameters
            */
        virtual void init_params() = 0;

        /*
            \brief Recover coordinates given unique cell id
            */
        virtual void get_cellCoord(int *ixyz, const int &i) const = 0;

        inline int get_nx() const { return Ns[0]; }

        inline int get_ny() const { return Ns[1]; }
    };

    class LinkListOrtho2D : public LinkListOrtho {
    protected:
        inline int get_cellID0(int *ls) const {
            return ls[0] * Ns[1] + ls[1];
        }

    public:
        LinkListOrtho2D() : LinkListOrtho() { this->init_params(); }

        inline void init_params() {
            DIM = 2;
            NEIGHBORS = DNS2D.size() / DIM;
            DNS = DNS2D.data();
        }

        inline void get_cellCoord(int *ixyz, const int &i) const {
            assert(i >= 0 && i < Ncells);
            ixyz[0] = static_cast<int>(i / Ns[1]);
            ixyz[1] = i - Ns[1] * ixyz[0];
            assert(ixyz[0] >= 0 && ixyz[0] < Ns[0]);
            assert(ixyz[1] >= 0 && ixyz[1] < Ns[1]);
        }
    };

    class LinkListOrtho3D : public LinkListOrtho {
    protected:
        inline int get_cellID0(int *ls) const {
            return (ls[0] * Ns[1] + ls[1]) * Ns[2] + ls[2];
        }

    public:
        LinkListOrtho3D() : LinkListOrtho() { this->init_params(); }

        inline void init_params() {
            DIM = 3;
            NEIGHBORS = DNS3D.size() / DIM;
            DNS = DNS3D.data();
        }

        inline void get_cellCoord(int *ixyz, const int &i) const {
            assert(i >= 0 && i < Ncells);
            ixyz[0] = static_cast<int>(i / (Ns[1] * Ns[2]));
            int ii = i - ixyz[0] * Ns[1] * Ns[2];
            ixyz[1] = static_cast<int>(ii / Ns[2]);
            ixyz[2] = ii - ixyz[1] * Ns[2];
            assert(ixyz[0] >= 0 && ixyz[0] < Ns[0]);
            assert(ixyz[1] >= 0 && ixyz[1] < Ns[1]);
            assert(ixyz[2] >= 0 && ixyz[2] < Ns[2]);
        }

        inline int get_nz() const { return Ns[2]; }
    };
}


/*!
        Flag Waiving by Kevlin Henny
        C++ Workshop column in Application Development Advisor 6(3), April 2002
        http://www.two-sdg.demon.co.uk/curbralan/papers/FlagWaiving.pdf
        Usage:

        namespace STYLE{
        enum STYLE{OPT1, OPT2, OPT3, OPT4, ..., _SIZE_};
      }

      template<> struct enum_traits<STYLE::STYLE>: enum_traiter<STYLE::STYLE, STYLE::_SIZE>{};
      enum_set<STYLE::STYLE> SW_STYLE;  // this is the variable containing all the possible "STYLE" options

      if(SW_STYLE[STYLE::OPT1]) ...

      */
template<typename type>
struct enum_traits {
    static const bool is_specialized = false;
    static const std::size_t count = 0;
};

template<typename type, type last_value>
struct enum_traiter {
    static const bool is_specialized = true;
    static const std::size_t count = (last_value - type()) + 1;
};

template<typename enum_type,
        typename traits = enum_traits<enum_type> >
class enum_set {
private:
    std::bitset <traits::count> bits;

public:
    enum_set() {
    }

    enum_set(enum_type setting) {
        set(setting);
    }

    std::size_t count() const {
        return bits.count();
    }

    std::size_t size() const {
        return bits.size();
    }

    bool operator[](enum_type testing) const {
        assert(testing >= 0 && testing < traits::count);
        return bits[testing];
    }

    enum_set operator~() const {
        return enum_set(*this).flip();
    }

    bool any() const {
        return bits.any();
    }

    bool any(std::initializer_list <enum_type> testlist) const {
        bool res = false;
        for (auto &testing : testlist) {
            assert(testing >= 0 && testing < traits::count);
            res = (res || bits[testing]);
        }
        return res;
    }

    bool none() const {
        return bits.none();
    }

    bool none(std::initializer_list <enum_type> testlist) const {
        bool res = false;
        for (auto &testing : testlist) {
            assert(testing >= 0 && testing < traits::count);
            res = (res || bits[testing]);
        }
        return !res;
    }

    bool all() const {
        return bits.all();
    }

    bool all(std::initializer_list <enum_type> testlist) const {
        bool res = true;
        for (auto &testing : testlist) {
            assert(testing >= 0 && testing < traits::count);
            res = (res && bits[testing]);
        }
        return res;
    }

    /*
        a  b  a AND b
        1  1   1
        1  0   0
        0  1   0
        0  0   0
        */
    bool any_and(std::initializer_list <enum_type> lista,
                 std::initializer_list <enum_type> listb) {
        return (any(lista) && any(listb));
    }

    /*
          a  b  a XOR b
          1  1   0
          1  0   1
          0  1   1
          0  0   0
          */
    bool any_xor(std::initializer_list <enum_type> lista,
                 std::initializer_list <enum_type> listb) {
        bool ta = any(lista);
        bool tb = any(listb);
        return (!(ta && tb) && (ta || tb));
    }

    /*
            a  b  a XNOR b (same as IF AND ONLY IF)
            1  1   1
            1  0   0
            0  1   0
            0  0   1
            */
    bool any_iff(std::initializer_list <enum_type> lista,
                 std::initializer_list <enum_type> listb) {
        bool ta = any(lista);
        bool tb = any(listb);
        return ((ta && tb) || (!ta && !tb));
    }

    /*
              a  b  a OR b
              1  1   1
              1  0   1
              0  1   1
              0  0   0
              */
    bool any_or(std::initializer_list <enum_type> lista,
                std::initializer_list <enum_type> listb) {
        return (any(lista) || any(listb));
    }

    enum_set &operator&=(const enum_set &rhs) {
        bits &= rhs.bits;
        return *this;
    }

    enum_set &operator|=(const enum_set &rhs) {
        bits |= rhs.bits;
        return *this;
    }

    enum_set &operator^=(const enum_set &rhs) {
        bits ^= rhs.bits;
        return *this;
    }

    enum_set &set() {
        bits.set();
        return *this;
    }

    enum_set &set(enum_type setting, bool value = true) {
        bits.set(setting, value);
        return *this;
    }

    enum_set &set(std::initializer_list <enum_type> setlist, bool value = true) {
        for (auto &setting : setlist) bits.set(setting, value);
        return *this;
    }

    enum_set &reset() {
        bits.reset();
        return *this;
    }

    enum_set &reset(enum_type resetting) {
        bits.reset(resetting);
        return *this;
    }

    enum_set &reset(std::initializer_list <enum_type> resetlist) {
        for (auto &resetting : resetlist) bits.reset(resetting);
        return *this;
    }

    enum_set &flip() {
        bits.flip();
        return *this;
    }

    enum_set &flip(enum_type flipping) {
        bits.flip(flipping);
        return *this;
    }

    enum_set &flip(std::initializer_list <enum_type> fliplist) {
        for (auto &flipping : fliplist) bits.flip(flipping);
        return *this;
    }
};

#endif
