#ifndef MD_HPP
#define MD_HPP

#include "Array.h"
#include "Common.hpp"
#include "Units.hpp"
#include "Random.hpp"
#include "Cell.hpp"
#include "Colony.hpp"
#include "Wall.hpp"
#include "Container.hpp"
#include "Input.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

/*! \file MD.hpp
 \brief Declare and partially implement forces between pairs of elements, pairs of cells, criteria for cell division, and integrator methods
 */

/*!
 \brief Base class for the definition of pair forces
 */
class PairForceParams {
protected:
    array2<double> rcut2;
public:
    /// Initialize and scale interaction diameters and energies for each pair of particles (i,j)
    virtual void init(const vector<ElemParams>& params, const array2<double>& rc2) = 0;

    /// Compute the force for a pair of particles (i,j) at distance r
    virtual bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, double& phi,
                           const bool& energy_record, double& potential_energy) const = 0;

    virtual bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, const double& sigi,
                           const double& sigj) const = 0;

    /// Reports the natural time scale of the interaction
    virtual double getTimeScale(const ElemParams& ei, const ElemParams& ej) const = 0;

    /// Get the cut off of the potential, i.e. the largest distance for which the potential is possibly non-zero.
    virtual double getRc2(const ElemParams& ei, const ElemParams& ej) const = 0;
};

/*!
 \brief Intercellular force for Lennard-Jones particles.
 */
class LJParams : public PairForceParams {
private:
    /// Computes the bare force modulus, before any force shift or direction is applied
    inline __host__ __device__ double elemForce0(const double& sij2, const double& r2) const {
        double ir2 = 1.0 / r2;               // 1 / r^2
        double sr2n = sij2 * ir2;             // (sigma / r)^2
        double srn = sr2n * sr2n * sr2n;         // (sigma / r)^n  [n=6]
        sr2n = srn * srn;                       // (sigma / r)^2n
        return ((srn - 2.0 * sr2n) * ir2);        // -(2(sigma/r)^12 - (sigma/r)^6) / r^2
    }
    /// Computes the potential energy of interactions forces.
    inline __host__ __device__ double energyPotential(const double& sij2, const double& r2) const {
        double ir2 = 1.0 / r2;               // 1 / r^2
        double sr2n = sij2 * ir2;             // (sigma / r)^2
        double srn = sr2n * sr2n * sr2n;         // (sigma / r)^n  [n=6]
        sr2n = srn * srn;
        double phi;
        phi = 4 * (sr2n - srn);
        return phi;
    }

public:
    array2<double> epsilon;                 // epsilon_ij = 24*sqrt(epsilon_ii * epsilon_jj)
    void init(const vector<ElemParams>& params, const array2<double>& rc2);
    /*!
        Compute LJ force between element pairs with energy recording
    */
    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, double& phi,
        const bool& energy_record, double& potential_energy) const {
        const int& i = ei.type;
        const int& j = ej.type;
        const double sij2 = SQ(0.5 * (ei.sigmaCoreX + ej.sigmaCoreX));
        const double rc2 = sij2 * rcut2(i, j);
        fij = 0.0;
        if (r2 <= rc2) {
            fij = elemForce0(sij2, r2) - (SWITCH[OPT::FORCE_SHIFTED] ? elemForce0(sij2, rc2) * sqrt(rc2 / r2) : 0.0);
            fij *= epsilon(i, j);
            if (energy_record) {
                phi = energyPotential(sij2, r2);
                phi *= epsilon(i, j);
                potential_energy += phi;
            }
            return true;
        }
        else {
            return false;
        }
    }
    /*!
        Compute LJ force between element pairs
    */
    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, const double& sigi,
        const double& sigj) const {
        const int& i = ei.type;
        const int& j = ej.type;
        const double sij2 = SQ(0.5 * (ei.sigmaCoreX + ej.sigmaCoreX));
        const double rc2 = sij2 * rcut2(i, j);
        fij = 0.0;
        if (r2 <= rc2) {
            fij = elemForce0(sij2, r2) - (SWITCH[OPT::FORCE_SHIFTED] ? elemForce0(sij2, rc2) * sqrt(rc2 / r2) : 0.0);
            fij *= epsilon(i, j);
            return true;
        }
        else {
            return false;
        }
    }
    /*!
     Provides an estimate of the natural time scale of the potential
     */
    inline __host__ __device__ double getTimeScale(const ElemParams& ei, const ElemParams& ej) const {
        double sij = 0.5 * (ei.sigmaCore + ej.sigmaCore);
        double lmin = 0.5 * sij;
        double fmax = 24.0 * epsilon(ei.type, ej.type) / sij;
        return lmin / fmax;
    }

    inline __host__ __device__ double getRc2(const ElemParams& ei, const ElemParams& ej) const {
        return rcut2(ei.type, ej.type) * SQ(0.5 *
            (ei.sigmaCore +
                ej.sigmaCore));
    }
};

/*!
 \brief Intercellular force for soft-core (SC) particles.
 */
class SCParams : public PairForceParams {
private:
    inline __host__ __device__ double tprime(const double& y, const double& xi) const {
        return 6.0 * y * (-y + xi) / (xi * xi * xi);
    }

    inline __host__ __device__ double t_fun(const double& y, const double& xi) const {
        return y * y * (3 * xi - 2 * y) / (xi * xi * xi);
    }

    inline __host__ __device__ double elemForce0(const int& i, const int& j, const double& sigCore, const double& r2) const {
        const double r = sqrt(r2);
        const double y = r - sigCore;
        const double& xi = sigmaWell(i, j);
        double fij = 0;

        if (y <= -xi) {
            fij = 0.0;
        }
        else if (-xi <= y && y < 0.0) {
            fij = -(epsilonCore(i, j) + epsilonWell(i, j)) * tprime(-y, xi) / r;
        }
        else if (0 <= y && y < xi) {
            fij = epsilonWell(i, j) * tprime(y, xi) / r;
        }
        else {
            fij = 0.0;
        }

        return fij;
    }

    inline __host__ __device__ double energyPotential(const int& i, const int& j, const double& sigCore, const double& r2) const {
        const double r = sqrt(r2);
        const double y = r - sigCore;
        const double& xi = sigmaWell(i, j);
        double phi = 0;

        if (y <= -xi) {
            phi = epsilonCore(i, j);
        }
        else if (-xi <= y && y < 0.0) {
            phi = (epsilonCore(i, j) + epsilonWell(i, j)) * t_fun(-y, xi) - epsilonWell(i, j);
        }
        else if (0 <= y && y < xi) {
            phi = epsilonWell(i, j) * t_fun(y, xi) - epsilonWell(i, j);
        }
        else {
            phi = 0;
        }
        return phi;
    }

public:
    //! epsilonCore is the maximum potential energy, as long as sigmaWell < 0.5*(ei.sigmaCoreX + ej.sigmaCoreX).
    array2<double> epsilonCore;  // E0 !!!
    //! epsilonWell is the minimum potential energy.
    array2<double> epsilonWell;  // Em
    //! sigmaWell determines the steepness of the potential and thus is a measure of the width of the well.
    array2<double> sigmaWell;    // xi
    void init(const vector<ElemParams>& params, const array2<double>& rc2);
    /*!
     Compute SC force between element pairs
     */
    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, double& phi,
        const bool& energy_record, double& potential_energy) const {
        const int& i = ei.type;
        const int& j = ej.type;
        const double sigCore = 0.5 * (ei.sigmaCoreX + ej.sigmaCoreX); // sigma, i.e. the position of the potential minimum
        const double rc2 = rcut2(i, j) * SQ(sigCore + sigmaWell(i, j));
        fij = 0.0;
        if (r2 > rcut2(i, j) * SQ(sigCore - sigmaWell(i, j)) && r2 <= rc2) {
            fij = elemForce0(i, j, sigCore, r2) -
                (SWITCH[OPT::FORCE_SHIFTED] ? elemForce0(i, j, sigCore, rc2) * sqrt(rc2 / r2) : 0.0);
            if (energy_record) {
                phi = energyPotential(i, j, sigCore, r2);
                potential_energy += phi;
            }
            return true;
        }
        else {
            return false;
        }
    }

    /*!
     Provides an estimate of the natural time scale of the potential
     */
    inline __host__ __device__ double getTimeScale(const ElemParams& ei, const ElemParams& ej) const {
        double lmin = 0.5 * sigmaWell(ei.type, ej.type);
        double fmax = epsilonCore(ei.type, ej.type) * Constants::PI / (2.0 * sigmaWell(ei.type, ej.type));
        return lmin / fmax;
    }

    inline __host__ __device__ double getRc2(const ElemParams& ei, const ElemParams& ej) const {
        double sigCore = 0.5 * (ei.sigmaCore + ej.sigmaCore);
        return rcut2(ei.type, ej.type) * SQ(sigCore + sigmaWell(ei.type, ej.type));
    }
};

/*!
 \brief Intercellular force for Hertzian contact mechanism particles.
 */
class HZParams : public PairForceParams {
private:
    inline __host__ __device__ double elemForce0(const ElemParams& ei, const ElemParams& ej, const double& r2, const double& sigi,
        const double& sigj) const {
        const int& i = ei.type;
        const int& j = ej.type;
        double fij = 0;
        const auto Ri = sigi / 2;
        const auto Rj = sigj / 2;
        const double vi = ei.sigmaWell;
        const double vj = ej.sigmaWell;
        const double Ei = ei.epsilonCore;
        const double Ej = ej.epsilonCore;
        const double fEl = pow(r2, 1.5) / ((0.75) * ((1 - SQ(vi)) / Ei + (1 - SQ(vj)) / Ej) * (sqrt((1 / Ri) + (1 / Rj))));
        return -fEl;
    }

public:
    //! epsilonCore is the maximum potential energy, as long as sigmaWell < 0.5*(ei.sigmaCoreX + ej.sigmaCoreX).
    array2<double> epsilonCore;  // elastic modulus
    //! epsilonWell is the minimum potential energy.
    array2<double> epsilonWell;  // adhesion constant
    //! sigmaWell determines the steepness of the potential and thus is a measure of the width of the well.
    array2<double> sigmaWell;    // poisson ratio
    void init(const vector<ElemParams>& params, const array2<double>& rc2);
    /*!
        Compute force between element pairs
    */
    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, double& phi,
        const bool& energy_record, double& potential_energy) const {
        fij = 0.0;
        fij = elemForce0(ei, ej, r2, 0, 0);
        return true;
    }

    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, const double& sigi,
        const double& sigj) const {
        fij = 0.0;
        fij = elemForce0(ei, ej, r2, sigi, sigj);
        return true;
    }
    /*!
    Provides an estimate of the natural time scale of the potential
    */
    inline __host__ __device__ double getTimeScale(const ElemParams& ei, const ElemParams& ej) const {
        double lmin = 0.5 * sigmaWell(ei.type, ej.type);
        double fmax = epsilonCore(ei.type, ej.type) * Constants::PI / (2.0 * sigmaWell(ei.type, ej.type));
        return lmin / fmax;
    }

    inline __host__ __device__ double getRc2(const ElemParams& ei, const ElemParams& ej) const {
        return rcut2(ei.type, ej.type) * SQ(0.5 * (ei.sigmaCore + ej.sigmaCore));
    }
};

/*!
Compute a simple elastic and attractive force between element pairs
*/
class EAParams : public PairForceParams {
private:
    inline __host__ __device__ double elemForce0(const ElemParams& ei, const ElemParams& ej, const double& r2, const double& sigi,
        const double& sigj) const {
        const int& i = ei.type;
        const int& j = ej.type;
        const double Ei = ei.epsilonCore;
        const double fEl = r2 * Ei;
        return -fEl;
    }

public:
    //! epsilonCore is the maximum potential energy, as long as sigmaWell < 0.5*(ei.sigmaCoreX + ej.sigmaCoreX).
    array2<double> epsilonCore;  // elastic modulus
    //! epsilonWell is the minimum potential energy.
    array2<double> epsilonWell;  // adhesion constant
    //! sigmaWell determines the steepness of the potential and thus is a measure of the width of the well.
    array2<double> sigmaWell;    // poisson ratio
    void init(const vector<ElemParams>& params, const array2<double>& rc2);
    /*!
        Compute LJ force between element pairs
    */
    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, double& phi,
        const bool& energy_record, double& potential_energy) const {
        fij = 0.0;
        fij = elemForce0(ei, ej, r2, 0, 0);
        return true;
    }

    inline __host__ __device__ bool elemForce(const ElemParams& ei, const ElemParams& ej, const double& r2, double& fij, const double& sigi,
        const double& sigj) const {
        fij = 0.0;
        fij = elemForce0(ei, ej, r2, sigi, sigj);
        return true;
    }
    /*!
    Provides an estimate of the natural time scale of the potential
    */
    inline __host__ __device__ double getTimeScale(const ElemParams& ei, const ElemParams& ej) const {
        double lmin = 0.5 * sigmaWell(ei.type, ej.type);
        double fmax = epsilonCore(ei.type, ej.type) * Constants::PI / (2.0 * sigmaWell(ei.type, ej.type));
        return lmin / fmax;
    }

    inline __host__ __device__ double getRc2(const ElemParams& ei, const ElemParams& ej) const {
        double sigCore = (ei.sigmaCore);
        return rcut2(ei.type) * SQ(2 * sigCore);
    }

};

struct MDParams {
    double dt;     // time step
    double hdt;    // half time step
    int frames;    // max frames
    int gts;       // printout interval
    double temp;   // environmental temperature
    double kbt;    // Thermal energy of system, = kB*T
    bool drain;    // Boolean for open boundary
    double pi;     // Osmotic pressure
    double tau_ve; // Vescolsticelastic response time

    double dissamcel; // Dissipative parameter of same cell elements.
    double disdifcel; // Dissipative parameter of different cell elements.
    double disbg;     // Dissipative parameter of between cells and background.
};

extern MDParams paramsMD;
extern WallParams paramsWall;
extern PairForceParams* paramsForce;
extern Container container;
extern CellColony colony;


/*!
  Intra-cellular forces for single cells : just Motility forces
*/
void singleCellForce(const CellParams& params, const int& state, const array2<double>& elemPos,
    const array2<double>& elemFrc, const array2<double>& elemVel, const array1<double>& Wiener,
    const array1<double>& sig);

void singleCellDivision(const double& rr, const double& sigmaTheta, const double& sigmaCore, const array2<double>& motherPos,
    const array2<double>& daughterPos);

/*!
  Intra-cellular forces for binary cells : 1 FENE bond + 2 Motility forces
*/
void binaryCellForce(const CellParams& params, const int& state, const array2<double>& elemPos,
    const array2<double>& elemFrc, const array2<double>& elemVel, const array1<double>& Wiener,
    const array1<double>& sig);

void binaryCellDivision(const double& rr, const double& sigmaTheta, const double& sigmaCore, const array2<double>& motherPos,
    const array2<double>& daughterPos);

void binaryCellDivisionFail(const CellParams& params, const array2<double>& motherPos);

/*!
 Compute the inter-cellular force for one cell-cell pair : LJ or soft core interactions between elements
 */
void cellCellForce(const CellParams& paramsi, const CellParams& paramsj, const cellIDs& ci, const cellIDs& cj, const Array::array2<double>& ri,
    const Array::array2<double>& rj, Array::array2<double>& fi, Array::array2<double>& fj);

void cellCellForcePressure(const CellParams& paramsi, const CellParams& paramsj, const cellIDs& ci, const cellIDs& cj, const Array::array2<double>& ri,
    const Array::array2<double>& rj, Array::array2<double>& fi, Array::array2<double>& fj, double& pri,
    double& prj);

void cellBackDisForce(const CellParams& params, const cellIDs& ci, const Array::array2<double>& ri,
    Array::array2<double>& fi, Array::array2<double>& vi);

void sameCellDisForce(const CellParams& params, const cellIDs& ci, const Array::array2<double>& ri,
    Array::array2<double>& fi, Array::array2<double>& vi);

void cellCellDisForce(const CellParams& params, const cellIDs& ci, const cellIDs& cj, const Array::array2<double>& ri,
    const Array::array2<double>& rj, Array::array2<double>& fi, Array::array2<double>& fj,
    Array::array2<double>& vi, Array::array2<double>& vj);

void sameCellDisRanForce(const CellParams& params, const cellIDs& ci, const Array::array2<double>& ri,
    Array::array2<double>& fid, Array::array2<double>& fir, Array::array2<double>& vi,
    Array::array1<double>& sigi);

void cellCellDisRanForce(const CellParams& params, const cellIDs& ci, const cellIDs& cj, const Array::array2<double>& ri,
    const Array::array2<double>& rj, Array::array2<double>& fid, Array::array2<double>& fjd,
    Array::array2<double>& fir, Array::array2<double>& fjr, Array::array2<double>& vi,
    Array::array2<double>& vj);

inline bool singleCellDivisionCriteria(const CellParams& params, const array2<double>& elemPos, const array1<double>& sigmaCore) {
    assert(params.numElems == 1 && elemPos.Nx() == 1);
    assert(elemPos.Ny() == DIM);
    return sigmaCore[0] >= params.cycleSigmaDiv;
}

inline bool binaryCellDivisionCriteria(const CellParams& params, const array2<double>& elemPos, const array1<double>& sigmaCore) {
    assert(params.numElems == 2 && elemPos.Nx() == 2);
    assert(elemPos.Ny() == DIM);
    double rij[DIM];
    return (params.divisionR2minRatio ? container.box.distance(rij, elemPos[1], elemPos[0]) >=
        params.divisionR2minRatio * sigmaCore[0] :
        container.box.distance(rij, elemPos[1], elemPos[0]) >= params.divisionR2min);
}
inline int randomTimeSpan(const Gaussian& rnd) {
    return MAX(Nint(Random::normal(rnd.mu, rnd.sigma) / paramsMD.dt), 1);
}

inline int randomTimeSpanLogNormal(const Gaussian& rnd) {
    return MAX(Nint(Random::logNormal(rnd.mu, rnd.sigma) / paramsMD.dt), 1);
}
inline int randomTimeSpanNonNegative(const Gaussian& rnd) {
    return true;
}



/*!
  MD Integrators
 */
typedef void(*MDIntegrator)(int& ts);

extern MDIntegrator mdStep;

void Euler(int& ts);

void StochasticHeun(int& ts);

void DPD(int& ts);

#endif
