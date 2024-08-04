#ifndef CELL_HPP
#define CELL_HPP

#include "Units.hpp"

struct cellIDs {
	int type;
	int id;

	cellIDs() : type(0), id(0) {}
	cellIDs(const int &_type, const int &_id) : type(_type), id(_id) {}
	bool operator==(const cellIDs &c) const { return (type == c.type && id == c.id); }
	bool operator!=(const cellIDs &c) const { return !((*this) == c); }
};

struct elemIDs {
	cellIDs cell;
	int type;
	int id;

	elemIDs() : cell(), type(0), id(0) {}

	elemIDs(const int &_cellType, const int &_cellID, const int &_elemType, const int &_elemID) :
			cell(_cellType, _cellID), type(_elemType), id(_elemID) {}
};
namespace CellState {
	enum CellState {
		Dividing = 0, Crawling = 1, Undefined = 2
	};
};

struct ElemParams {
	int type;
	double m;
	double growthk0;
	double growthk1;
	double growthk2;
	double zeta;
	double zeta_p;
	double zeta_p_beta;
	double static_vel;
	double static_force;
	double ms;
	double zetaInv;
	double rodRadius;
	double sigmaCore;
	double sigmaTheta;
	double sigmaMin;
	double sigmaMax;
	double sigmaWell;
	double elasticModulus;
	double poissonRatio;
	double adhesionConstant;
	double epsilonCore;
	double epsilonWell;
	double sigmaCoreX;

	inline __host__ __device__ void updateCore(const double &tmpSigma) { sigmaCoreX = tmpSigma; }
};

struct CellParams {                    // Cell parameters
	bool CIL;                 // CIL on or off?
	bool DIV;                 // Can the cell divide?
	bool APO;                 // Can the cell die?
	bool EXTERNAL;            // Constant exernal driving force?

	double crossAdhesion;       // The strength of surface tension of one kind colony/tissue again other, to replace the previous cross-adhesion-scaling.
	bool shape;            // 0 represents circle, 1 represents rod;
	int numElems;            //number of elements
	int numBonds;            //number of inner bonds
	array1<int> elemType;            //unique element ids
	array1<double> m;                   //m_i for all i in in species_elements

	// During the division phase, the motility force of cell i to drive cells at distance r apart is given by 
	//    activity[i] = (-params.growthk0[i] * feneDivMax + params.growthk1[i] + params.growthk2[i] * SQ(r2));
	array1<double> growthk0;            //g_i for all i in in species_elements
	array1<double> growthk1;            //g_i for all i in in species_elements
	array1<double> growthk2;            //g_i for all i in in species_elements

	// Friction parameters: The friction coefficient of cell i in its most general form is given as a function of the nondimensionalised pressure p/p0
	//    params.zeta(i) + params.zeta_p(j) * pow(p/p0, params.zeta_p_beta(j))
	// TODO It seems that zetaInv is currently incorrectly used as the inverse of that whole expression but only of the constant zeta
	array1<double> zeta;                //inverse constant friction coefficient zeta for all i in species_elements
	array1<double> zeta_p;              //...
	array1<double> zeta_p_beta;         //...
	array1<double> static_vel;
	array1<double> static_force;
	array1<double> ms;                  //ms_i for all i in in species_elements
	array1<double> zetaInv;							// inverse of the inverse friction coefficient (SKS: I know this is insane)

	array1<double> rodRadius;           //for rod-like shape cells. 

	map<string, int> mapNames;          //element names

	// single element cell interaction parameters for all i in species_elements
	array1<double> sigmaCore;           //sigma_ii    (LJ & SC & HZ)
	array1<double> sigmaTheta;          //theta, cell cycle progress.
	// array1<double> length;              //for rod-like shape cells.
	array1<double> sigmaMin;            //min diameter
	array1<double> sigmaMax;            //max diameter
	array1<double> sigmaWell;           //sigma0_ii   (SC) = xi , (HZ) = Poisson ratio
	array1<double> epsilonCore;         //epsilon0_ii (SC) , (HZ) = elastic modulus
	array1<double> epsilonWell;         //epsilon_ii  (LJ & SC), (HZ) = adhesion constant

	// single element cell noise parameters
	array1<double> dNoise;              //passive/radial/angular noise amplitudes for cell
	array1<double> fExt;

	// FENE parameters for inner bonds between pairs of elements
	array2<double> kappa;               //kappa_ij
	array2<double> kappa_push;          //for rod-like cells.
	array2<double> spring_ratio;        //for rod-like cells.
	array2<double> r2max;               //rmax_ij: square of maximum extension of FENE springs between any given two elements
	array2<double> divisionR2max;       //rmax_ij during division
	array2<double> rss;                 //single-particle steady state distances

	// Cell division parameters
	Gaussian migrationTau;        //migration time
	Gaussian divisionTau;         //division time
	// double divisionPressure;    //threshold pressure decides cell cycle if continues. The primitive version of cell cycle. UNUSED IN CODE
	double divisionR0;          //reset distance for division
	double divisionR2min;       //division distance criteria
	double divisionR2minRatio;
	double divisionSigmaBirthRatio;
	double divisionSigmaMin;
	bool divisionSwelling;      // determines whether the smaller elements swell up during division
	bool divisionConstForce;
	bool typeswitching;         // determines whether the cells switch type when they divide

	// Cell cycle parameters, see cell cycle equations for details
	double v_s;									// volume of the senescent/quiescent cell.
	double v_1;                 // mean-field maximum unstressed volume of a cell in units of the quiescent volume v_s
	// double cycleGamma;          // cell cycle volume adjustment parameter, ~ to \alpha. Assume 1 for now.
	double cycleTauR;           // cell cycle relaxing time.
	double cycleTauV;           // cell volume relaxing time.
	double cycleSigmaDiv;       // minimal diameter for cell division. Assume 0 for now, means no limitation.
	// Following three parameters do not have effects now. It's designed for 3D MCS shrinkage.
	double cycleTauP;						// direct pressure impact-cell volume relaxing time. For physical quick shrinkage. SKS: as I understand it, if this is non-zero, cell volumes will be reduced by high pressures w.r.t a target pressure. If cycleTauP == 0, then the cell volumes will always be at target pressure. CANDIDATE FOR REMOVAL.
	double cycleP0;             // reference pressure for cell cycle regulation, and physical cell shrinkage if cycleTauP nonzero.
	double cyclePShift; // pressure shift due to traction with the substrate, in units of cycleP0. Cannot make the pressure negative by itself. FEATURE UNDER DEVELOPMENT
	double cyclePShiftTauInv; // inverse of the time on which the pressure shift decays.

	double cycleRhoRatio;       // asymmetric cell cycle arrest/recover ratio.  CANDIDATE FOR REMOVAL.
	double cycleRMaxedInv;			// Inverse of the r-value for which the cell cycle velocity is maximal. The cell cycle regulation used in Li et al, PRX (2021) corresponds to a value of 1, which is used by default.
	double betaVs;              // quick shrinkage parameter. CANDIDATE FOR REMOVAL.

	// Cell apoptosis parameters
	double apopPressure;        // the relatively high pressure for cells
	double apopDuration;        // the duration for cells to stay under high pressure before apoptosis
	double apopRate;            // the maxmium apoptosis rate
	double apopAlpha;           // the apoptosis activity of apoptosis calculation
	double apopGamma;           // the exponential coefficient of apoptosis calculation
	double extrRate;            // the maximum extrusion rate, published as $k^{(0)}$
	double extrGamma;           // the exponential coefficient of extrusion calculation

	// DPD parameters
	double dissamcel;           // dissipation parameter between two elements within the same cell.
	double disdifcel;           // dissipation parameter between elements of the different cell.
	double disbg;               // dissipation parameter between cell and background.
	double kbtn;                // Noise intensity in the tissue.

	void init(int &_id0, const int _numElems) {
		run_assert(_numElems > 0);
		numElems = _numElems;
		numBonds = numElems * (numElems - 1) / 2;

		CIL = DIV = APO = EXTERNAL = false;

		// unique id for element types
		elemType.Allocate(numElems);
		for (auto i : irange(0, numElems)) elemType[i] = _id0 + i;
		m.Allocate(numElems);
		m = 0.0;
		growthk0.Allocate(numElems);
		growthk0 = 0.0;
		growthk1.Allocate(numElems);
		growthk1 = 0.0;
		growthk2.Allocate(numElems);
		growthk2 = 0.0;
		zeta.Allocate(numElems);
		zeta = 0.0;
		zeta_p.Allocate(numElems);
		zeta_p = 0.0;
		zeta_p_beta.Allocate(numElems);
		zeta_p_beta = 0.0;
        static_vel.Allocate(numElems);
        static_vel = 0.0;
        static_force.Allocate(numElems);
        static_force = 0.0;
		ms.Allocate(numElems);
		ms = 0.0;
		zetaInv.Allocate(numElems);
		zetaInv = 0.0;
		rodRadius.Allocate(numElems);
		rodRadius = 0.0;

		sigmaCore.Allocate(numElems);
		sigmaCore = 0.0;
		sigmaTheta.Allocate(numElems);
		sigmaTheta = 0.0;
		sigmaMin.Allocate(numElems);
		sigmaMin = 0.0;
		sigmaMax.Allocate(numElems);
		sigmaMax = 0.0;
		sigmaWell.Allocate(numElems);
		sigmaWell = 0.0;
		epsilonCore.Allocate(numElems);
		epsilonCore = 0.0;
		epsilonWell.Allocate(numElems);
		epsilonWell = 0.0;

		dNoise.Allocate(NOISE::_SIZE_);
		dNoise = 0.0;
		fExt.Allocate(DIM);
		fExt = 0.0;

		kappa.Allocate(numElems, numElems);
		kappa = 0.0;
		kappa_push.Allocate(numElems, numElems);
		kappa_push = 0.0;
		spring_ratio.Allocate(numElems, numElems);
		spring_ratio = 0.0;
		r2max.Allocate(numElems, numElems);
		r2max = 0.0;
		divisionR2max.Allocate(numElems, numElems);
		divisionR2max = 0.0;
		rss.Allocate(numElems, numElems);
		rss = 0.0;
		_id0 += _numElems;

		migrationTau.mu = migrationTau.sigma = 0.0;
		divisionTau.mu = divisionTau.sigma = 0.0;
		// divisionPressure = 0.0;
		divisionR0 = 0.0;
		divisionR2min = 0.0;
		divisionR2minRatio = 0.0;
		divisionSigmaBirthRatio = 0.0;
		divisionSigmaMin = 0;
		divisionSwelling = false;
		// cycleAlpha = 0.0;
		// cycleBeta = 0.0;
		v_1 = 1.0;
		// cycleGamma = 0.0;
		cycleTauR = 0.0;
		cycleTauV = 0.0;
		cycleTauP = 0.0;
		cycleSigmaDiv = 0.0;
		v_s = 0.0;
		cycleP0 = 0.0;
		cyclePShift = 0.0;
		cyclePShiftTauInv = 0.0;
		betaVs = 0.0;
		cycleRhoRatio = 1.0;
		cycleRMaxedInv = 1.0;

		shape = 0;
		crossAdhesion = 0.0;
		dissamcel = 0.0;
		disdifcel = 0.0;
		disbg = 0.0;
		kbtn = 0.0;
		apopDuration = 0.0;
		apopPressure = 0.0;
		apopRate = 0.0;
	}
};
#endif
