#include "OutputWriter.hpp"

extern double potentialEnergy, cpuTime;
static const int BUFFER_SIZE = 64;
double lz2d;
char buffer[BUFFER_SIZE];

std::ofstream lammpsOut;
SimIO::SimIO h5Out;
gsd_handle gsdOut;

void (*gsdBufferVectorShiftScale)(const int &, const array3<double> &, const array1<double> &, const array1<double> &,
																	float *);

void (*gsdBufferVectorScaleElem)(const int &, const array3<double> &, const array1<double> &, float *);

void (*gsdBufferVector)(const int &type, const array3<double> &, float *);

void (*gsdBufferQuaternion)(const int &type, const array1<double> &, float *);

/*!
\brief copy element parameter to all elements of given type
*/
template<typename T, typename Q>
inline void copyElementParameter(const int &type, array1<T> &elemData, Q *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = elemData.Size();
	Q *ptr = ptr0 + colony.cumulElems[type];
	for (auto i : irange(0, numCells))
		std::copy(elemData.begin(), elemData.end(), ptr + i * numElems);
}

/*!
\brief copy cell data to all elements of given cell
*/
template<typename T, typename Q>
inline void copyCellData(const int &type, array1<T> &cellData, Q *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array2<Q> dmy(numCells, numElems, ptr0 + colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) // broadcast cell data to all elements
		std::fill_n(&dmy[i][0], numElems, cellData[i]);
}

template<typename T, typename Q>
inline void copyTheta2Quaternion(const int &type, const array2<T> &cellScalar, Q *ptr0) {
//    std::cout<<"test"<<std::endl;
//    auto numCells = colony.getNumCells(type);
//    auto numElems = colony.paramCells[type].numElems;
//    std::cout<<"test01"<<std::endl;
//    array2<Q> dmy2d(numCells, 4*numElems, ptr0 + colony.cumulElems[type]);
//    std::cout<<"test1"<<std::endl;
//
//
//    for (auto i : irange(0, numCells)) {
//        for (auto j : irange(0, numElems)) {
//
//            auto thetaHalf=(cellScalar(i,0)/360)*Constants::PI;
//            auto theta=(cellScalar(i,0)/180)*Constants::PI;
//            auto tanSQ=SQ(tan(theta));
//            auto x=sqrt(0.5/(tanSQ+1));
//            auto y=tan(theta)*x;
//            auto quaternion = [1,x,y,0];
//            dmy2d(i, j) = quaternion;
//        }
//    }
//
//
//    for (auto i : irange(0, numCells)) // broadcast cell data to all elements
//    {
//        std::cout<<"test"<<std::endl;
//        auto thetaHalf=(cellScalar[i][0]/360)*Constants::PI;
//        auto theta=(cellScalar[i][0]/180)*Constants::PI;
//        auto tanSQ=SQ(tan(theta));
//        auto x=sqrt(0.5/(tanSQ+1));
//        auto y=tan(theta)*x;
//        std::fill_n(&dmy2d[i][0], numElems, 1);
//        std::fill_n(&dmy2d[i][0], numElems, x);
//        std::fill_n(&dmy2d[i][0], numElems, y);
//        std::fill_n(&dmy2d[i][0], numElems, 0);
//    }
}
/*!
\brief assign constant value to all elements of given cell type
*/
template<typename T, typename Q,
		typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline void copyCellData(const int &type, const T &val, Q *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	std::fill_n(ptr0 + colony.cumulElems[type], numCells * numElems, val);
}

/*!
\brief copy transformed cell data to all elements of given cell
*/
template<typename T, typename Q>
inline void copyCellData(const int &type, const array1<T> &cellData, std::function<Q(const T &t0)> &func, Q *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array2<Q> dmy(numCells, numElems, ptr0 + colony.cumulElems[type]);
	for (auto i : irange(0, numCells))
		std::fill_n(&dmy[i][0], numElems, func(cellData[i]));
}

/*!
\brief create ...
*/
template<typename T, typename Q>
inline void gsdBufferScalar(const int &type, const array2<T> &cellScalar, Q *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array2<Q> dmy2d(numCells, numElems, ptr0 + colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) {
		for (auto j : irange(0, numElems)) {
			dmy2d(i, j) = cellScalar(i, j);
		}
	}
}

/*!
\brief create 3d buffered data from 2d cell data
shift: shift individual components (needed to recenter box)
zval : missing z component per element
*/
inline void
gsdBufferVectorShiftScale2D(const int &type, const array3<double> &cellVector2d, const array1<double> &shift,
														const array1<double> &zval, float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, MAX_DIM, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) {
		for (auto j : irange(0, numElems)) {
			dmy3d(i, j, 0) = cellVector2d(i, j, 0) - shift[0];
			dmy3d(i, j, 1) = cellVector2d(i, j, 1) - shift[1];
			dmy3d(i, j, 2) = zval[j];
		}
	}
}

inline void
gsdBufferVectorShiftScale3D(const int &type, const array3<double> &cellVector3d, const array1<double> &shift,
														const array1<double> &zval, float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, MAX_DIM, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells))
		for (auto j : irange(0, numElems))
			for (auto d : irange(0, MAX_DIM))
				dmy3d(i, j, d) = cellVector3d(i, j, d) - shift[d];
}

/*!
\brief create 3d buffered data from 2d cell data
*/
inline void gsdBufferVector2D(const int &type, const array3<double> &cellVector2d, float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, MAX_DIM, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) {
		for (auto j : irange(0, numElems)) {
			dmy3d(i, j, 0) = cellVector2d(i, j, 0);
			dmy3d(i, j, 1) = cellVector2d(i, j, 1);
			dmy3d(i, j, 2) = 0.0;
		}
	}
}

inline void gsdBufferQuaternion2D(const int &type, const array1<double> &cellData, float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, 4, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) {
			auto theta= cellData(i);
			auto tanSQ= SQ(tan(theta));
			auto x=sqrt(0.5/(tanSQ+1));
			auto y=tan(theta)*x;

			dmy3d(i, 0, 0) = sqrt(0.5);
			dmy3d(i, 0, 1) = x;
			dmy3d(i, 0, 2) = y;
			dmy3d(i, 0, 3) = 0;
    }
}

inline void gsdBufferVector3D(const int &type, const array3<double> &cellVector3d, float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, MAX_DIM, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells))
		for (auto j : irange(0, numElems))
			for (auto d : irange(0, MAX_DIM))
				dmy3d(i, j, d) = cellVector3d(i, j, d);
}

inline void gsdBufferQuaternion3D(const int &type, const array1<double> &cellData, float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, 4, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) {
			dmy3d(i, 0, 0) = cellData(i);
			dmy3d(i, 0, 1) = cellData(i);
			dmy3d(i, 0, 2) = 0.0;
			dmy3d(i, 0, 3) = 0.0;
	}
}

/*!
\brief create 3d buffered data from 2d cell data
elemScale : scaling factor
*/
inline void
gsdBufferVectorScaleElem2D(const int &type, const array3<double> &cellVector2d, const array1<double> &elemScale,
													 float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, MAX_DIM, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells)) {
		for (auto j : irange(0, numElems)) {
			dmy3d(i, j, 0) = cellVector2d(i, j, 0) * elemScale[j];
			dmy3d(i, j, 1) = cellVector2d(i, j, 1) * elemScale[j];
			dmy3d(i, j, 2) = 0.0;
		}
	}
}

inline void
gsdBufferVectorScaleElem3D(const int &type, const array3<double> &cellVector3d, const array1<double> &elemScale,
													 float *ptr0) {
	auto numCells = colony.getNumCells(type);
	auto numElems = colony.paramCells[type].numElems;
	array3<float> dmy3d(numCells, numElems, MAX_DIM, ptr0 + MAX_DIM * colony.cumulElems[type]);
	for (auto i : irange(0, numCells))
		for (auto j : irange(0, numElems))
			for (auto d : irange(0, MAX_DIM))
				dmy3d(i, j, d) = cellVector3d(i, j, d) * elemScale[j];
}


void initHDF5Output(SimIO::SimIO &fp, const char *fileName) {
	fp.create_file(fileName);
	//fp.write_attr("version", VERSION);
	fp.create_group("params");
	// System Parameters
	{
		fp.write_attr("dt", paramsMD.dt);
		std::vector<double> ls;
		container.box.length(ls);
		fp.write_attr("L", ls.data(), DIM);
		fp.write_attr("dt", paramsMD.dt);
		fp.write_attr("Tdump", paramsMD.gts);
		fp.write_attr("Force", (SWITCH[OPT::FORCE_SOFTCORE] ? "SoftCore" : "LennardJones"));
	}
	{
		fp.create_group("units");
		fp.write_attr("length", units.Length);
		fp.write_attr("motility", units.Motility);
		fp.write_attr("friction", units.Friction);
		fp.write_attr("energy", units.Energy);
		fp.write_attr("time", units.Time);
		fp.write_attr("density", units.Density);
		fp.write_attr("force", units.Force);
		fp.write_attr("pressure", units.Pressure);
		fp.write_attr("Dspatial", units.SpatialDiffusion);
		fp.write_attr("Dangular", units.AngularDiffusion);
		fp.close_group();
	}

	// Cell Parameters
	{
		fp.create_group("cell");
		auto nTypes = colony.getCellTypes();
		double gaussParams[2];
		for (auto cellType : irange(0, nTypes)) {
			fp.create_group(colony.nameCells[cellType].c_str());
			auto &params = colony.paramCells[cellType];
			auto dimSpace = SimIO::Space::create(DIM);
			auto vectorSpace = SimIO::Space::create(params.numElems);
			auto matrixSpace = SimIO::Space::create(params.numElems, params.numElems);

			fp.write_attr("CIL", params.CIL);
			fp.write_attr("DIV", params.DIV);
			fp.write_attr("APO", params.APO);
			fp.write_attr("EXTERNAL", params.EXTERNAL);

			fp.write_attr("nElems", params.numElems);
			fp.write_attr("nBonds", params.numBonds);

			gaussParams[0] = params.migrationTau.mu;
			gaussParams[1] = params.migrationTau.sigma;
			fp.write_attr("migrationTau", gaussParams, 2);

			gaussParams[0] = params.divisionTau.mu;
			gaussParams[1] = params.divisionTau.sigma;
			fp.write_attr("divisionTau", gaussParams, 2);
			fp.write_attr("divisionR0", params.divisionR0);
			fp.write_attr("divisionRmin", sqrt(params.divisionR2min));

			fp.write_attr("apopRate", params.apopRate);

			fp.write_attr("Dpassive", SQ(params.dNoise[NOISE::PASSIVE]) * paramsMD.hdt);
			fp.write_attr("Dradial", SQ(params.dNoise[NOISE::RADIAL]) * paramsMD.hdt);
			fp.write_attr("Dangular", SQ(params.dNoise[NOISE::ANGULAR]) * paramsMD.hdt);

			fp.write_data("type", vectorSpace, params.elemType());
			fp.write_data("m", vectorSpace, params.m());
			fp.write_data("zeta", vectorSpace, params.zeta());
			fp.write_data("sigma_core", vectorSpace, params.sigmaCore());
			fp.write_data("sigma_max", vectorSpace, params.sigmaMax());
			fp.write_data("sigma_well", vectorSpace, params.sigmaWell());
			fp.write_data("epsilon_core", vectorSpace, params.epsilonCore());
			fp.write_data("epsilon_well", vectorSpace, params.epsilonWell());
			fp.write_data("kappa", matrixSpace, params.kappa());
			fp.write_data("r2max", matrixSpace, params.r2max());
			fp.write_data("rss", matrixSpace, params.rss());

			fp.write_data("fExt", dimSpace, params.fExt());

			SimIO::Space::close(dimSpace);
			SimIO::Space::close(vectorSpace);
			SimIO::Space::close(matrixSpace);
			fp.close_group();
		}
		fp.close_group();
	}

	// LJ Parameters
	/*if(SWITCH[OPT::FORCE_SOFTCORE]){
	fp.create_group("sc");
	fp.close_group();
}else if(!SWITCH[OPT::FORCE_GHOSTS]){
fp.create_group("lj");
auto paramsLJ = dynamic_cast<LJParams&>(*paramsForce);
auto matrixSpace = SimIO::Space::create(paramsLJ.sigma2.Nx(), paramsLJ.sigma2.Ny());
fp.write_data("sigma2", matrixSpace,  paramsLJ.sigma2());
fp.write_data("epsilon", matrixSpace, paramsLJ.epsilon());
fp.write_data("rcut2", matrixSpace,   paramsLJ.rcut2());
fp.close_group();
}*/
	fp.close_group();
	fp.create_group("/snapshots");
	fp.close_group();
	if (paramsOut.div_app_record) {
		fp.create_group("/div_app");
		fp.create_group("DIV");

		if (colony.getCellTypes() > 1) {
			for (auto cellType : irange(0, colony.getCellTypes())) {
				fp.create_group(colony.nameCells[cellType].c_str());
				fp.close_group();
			}
		}
		fp.close_group();
		fp.create_group("APP");
		if (colony.getCellTypes() > 1) {
			for (auto cellType : irange(0, colony.getCellTypes())) {
				fp.create_group(colony.nameCells[cellType].c_str());
				fp.close_group();
			}
		}
		fp.close_groups();
	}
}

/*!
 \brief Function for calculating the total kinetic energy of the system.
 */

double calculateKineticEnergy(const vector <array3<double>> &velocities) {
	double ke = 0.0;
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &params = colony.paramCells[ispec];
		const auto nCells = colony.getNumCells(ispec);
		auto &vel = velocities[ispec];
#pragma omp parallel for
		for (auto i = 0; i < nCells; i++) {
			const auto &vi = vel[i];
			for (auto j : irange(0, params.numElems)) {
				for (auto d : irange(0, DIM)) {
					ke += 0.5 * SQ(vi[j][d]);
				}
			}
		}
	}
	return ke;
}

void outputHDF5Frame(SimIO::SimIO &fp, const int &id, const int &ts) {
	// Initialize

	if (id >= 0) {
		fp.open_group("/snapshots");
		fp.create_group(("t_" + std::to_string(id)).c_str());
		fp.write_attr("ts", ts);
	}
	// Write species data
	for (auto cellType : irange(0, colony.getCellTypes())) {
		snprintf(buffer, BUFFER_SIZE, "%s", colony.nameCells[cellType].c_str());
		auto &params = colony.paramCells[cellType];
		auto nCells = colony.getNumCells(cellType);
		auto const PE = potentialEnergy;
		auto const CT = cpuTime;
		double KE;
		if (paramsOut.energy_record) {
			//Calculating energy information and writting it into file.
			KE = calculateKineticEnergy(colony.vel);
		}
		{// Write H5 data
			if (colony.getCellTypes() > 1) fp.create_group(colony.nameCells[cellType].c_str());
			auto vectorSpace = SimIO::Space::create(nCells, params.numElems, DIM);
			auto scalarSpace = SimIO::Space::create(nCells, params.numElems);
			auto energySpace = SimIO::Space::create(1);
			auto scalarCellSpace = SimIO::Space::create(nCells);
			fp.write_data("R", vectorSpace, colony.pos[cellType]());
			fp.write_data("Vel", vectorSpace, colony.vel[cellType]());
			fp.write_data("SigmaCore", scalarSpace, colony.sigmaCore[cellType]());
			fp.write_data("F", vectorSpace, colony.frc[cellType]());
			fp.write_data("FD", vectorSpace, colony.frcDis[cellType]());
			fp.write_data("FR", vectorSpace, colony.frcRan[cellType]());
			fp.write_data("RHO", scalarCellSpace, colony.cellTRho[cellType]());
			fp.write_data("Theta", scalarCellSpace, colony.cellTTheta[cellType]());
			fp.write_data("PR", scalarCellSpace, colony.pressure[cellType]());
			fp.write_data("State", scalarCellSpace, colony.cellState[cellType]());
			fp.write_data("T0", scalarCellSpace, colony.cellT0[cellType]()); // cell age = ts - t0
			fp.write_data("CpuTime", energySpace, &CT);
			if (paramsOut.energy_record) {
				fp.write_data("PotentialEnergy", energySpace, &PE);
				fp.write_data("KineticEnergy", energySpace, &KE);
			}
			SimIO::Space::close(vectorSpace);
			SimIO::Space::close(scalarSpace);
			SimIO::Space::close(scalarCellSpace);
			SimIO::Space::close(energySpace);
			if (colony.getCellTypes() > 1) fp.close_group();
		}
	}
	// Finalize
	if (id >= 0) fp.close_groups();
	if (paramsOut.div_app_record) {
		if (id >= 0) {
			fp.open_group("/div_app");
			fp.open_group("DIV");
			for (auto cellType : irange(0, colony.getCellTypes())) {
				snprintf(buffer, BUFFER_SIZE, "%s", colony.nameCells[cellType].c_str());
				auto &params = colony.paramCells[cellType];
				auto nCellsDiv = colony.getNumCellsDiv(cellType);
				auto cumCellsDiv = colony.getCumCellsDiv(cellType);
				{// Write H5 data
					if (colony.getCellTypes() > 1) fp.open_group(colony.nameCells[cellType].c_str());
					if ((nCellsDiv || params.DIV) == 0) {
						if (colony.getCellTypes() > 1) fp.close_group();
						continue;
					} else {
						//for (int i : irange(cumCellsDiv - nCellsDiv, cumCellsDiv)) {
							fp.create_group(("d_" + std::to_string(id)).c_str());
							auto vectorSpace = SimIO::Space::create(nCellsDiv, params.numElems, DIM);
							auto scalarCellSpace = SimIO::Space::create(nCellsDiv);
							fp.write_data("R", vectorSpace, colony.posCellsDiv[cellType]());
							fp.write_data("T", scalarCellSpace, colony.cellT0Div[cellType]());
							SimIO::Space::close(vectorSpace);
							SimIO::Space::close(scalarCellSpace);
							fp.close_group();
						//}
					}
					if (colony.getCellTypes() > 1) fp.close_group();
				}
				colony.resetNumCellsDiv(cellType);
			}
			fp.close_group();
			fp.open_group("APP");

			for (auto cellType : irange(0, colony.getCellTypes())) {
				snprintf(buffer, BUFFER_SIZE, "%s", colony.nameCells[cellType].c_str());
				auto &params = colony.paramCells[cellType];
				auto nCellsApp = colony.getNumCellsApp(cellType);
				auto cumCellsApp = colony.getCumCellsApp(cellType);

				{// Write H5 data
					if (colony.getCellTypes() > 1) fp.open_group(colony.nameCells[cellType].c_str());
					if ((nCellsApp || params.APO) == 0) {
						if (colony.getCellTypes() > 1) fp.close_group();
						continue;
					} else {
//						for (int i : irange(cumCellsApp - nCellsApp, cumCellsApp)) {
							fp.create_group(("a_" + std::to_string(id)).c_str());
							auto vectorSpace = SimIO::Space::create(nCellsApp, params.numElems, DIM);
							auto scalarCellSpace = SimIO::Space::create(nCellsApp);
							fp.write_data("R", vectorSpace, colony.posCellsApp[cellType]());
							fp.write_data("T", scalarCellSpace, colony.cellT0App[cellType]());
							SimIO::Space::close(vectorSpace);
							SimIO::Space::close(scalarCellSpace);
							fp.close_group();
			//			}
					}
					if (colony.getCellTypes() > 1) fp.close_group();
				}
				colony.resetNumCellsApp(cellType);
			}
			fp.close_group();
		}
	}
	fp.close_groups();
}

inline void outputGSDConf(gsd_handle &fp, const int &ts) {
	double lz = (DIM == 3 ? container.box.lz() : lz2d);
	uint64_t _ts = ts;
	uint8_t dim = MAX_DIM;
	float box[6] = {static_cast<float>(container.box.lx()),
									static_cast<float>(container.box.ly()),
									static_cast<float>(lz),
									0.0, 0.0, 0.0};

	run_assert(gsd_write_chunk(&fp, "configuration/step", GSD_TYPE_UINT64, 1, 1, 0, &_ts) == 0,
						 "write configuration/step");
	run_assert(gsd_write_chunk(&fp, "configuration/dimensions", GSD_TYPE_UINT8, 1, 1, 0, &dim) == 0,
						 "write configuration/dimensions");
	run_assert(gsd_write_chunk(&fp, "configuration/box", GSD_TYPE_FLOAT, 6, 1, 0, &box) == 0,
						 "write configuration/box");
}

void outputGSDParticles(gsd_handle &fp, const int &ts) {
	const auto totalElems = colony.getTotalElems();
	const auto cellTypes = colony.getCellTypes();
	float *fptr = colony.getVectorBuffer();
	unsigned int *uptr = colony.getScalarUnsignedBuffer();
	array1<double> hbox(DIM);
	container.box.length(hbox());
	for (auto d : irange(0, DIM)) hbox[d] *= 0.5;

	{ // Total number of elements / particles
		uint32_t n = totalElems;
		run_assert(gsd_write_chunk(&fp, "particles/N", GSD_TYPE_UINT32, 1, 1, 0, &n) == 0,
							 "write particles/N");
	}

	if (ts == 0) { // Write type/particle names
		const int maxSize = 63;
		int nTypes = colony.getElemTypes();
		char types[nTypes * maxSize];
		for (auto i : irange(0, nTypes * maxSize)) types[i] = 0;

		for (auto cellType : irange(0, colony.getCellTypes())) {
			auto &params = colony.paramCells[cellType];
			if (params.numElems > 1) {
				for (auto &keyval : params.mapNames) {
					auto &etag = keyval.first;
					auto &eid = keyval.second;

					snprintf(types + params.elemType[eid] * maxSize, maxSize, "%s_%s",
									 colony.nameCells[cellType].c_str(), etag.c_str());
				}
			} else {
				snprintf(types + params.elemType[0] * maxSize, maxSize, "%s",
								 colony.nameCells[cellType].c_str());
			}
		}
		run_assert(gsd_write_chunk(&fp, "particles/types", GSD_TYPE_INT8, nTypes, maxSize, 0, types) == 0,
							 "write particles/types");
	}

	if (SWITCH[OPT::CELLS_DIVIDING] || SWITCH[OPT::CELLS_DYING] || ts == 0) { // Constant data for cases in which the number of cells is constant, i.e. for non-dividing and non-dying cells
		for (auto cellType : irange(0, cellTypes))
			gsdBufferScalar(cellType, colony.sigmaCore[cellType], fptr);
            run_assert(gsd_write_chunk(&fp, "particles/Z", GSD_TYPE_FLOAT, totalElems, 1, 0, fptr) == 0,
                         "write particles/Z");

        for (auto cellType : irange(0, cellTypes))
            gsdBufferScalar(cellType, colony.sigmaCore[cellType], fptr);
//        run_assert(gsd_write_chunk(&fp, "particles/X", GSD_TYPE_FLOAT, totalElems, 1, 0, fptr) == 0,
//                   "write particles/X");
		run_assert(gsd_write_chunk(&fp, "particles/diameter", GSD_TYPE_FLOAT, totalElems, 1, 0, fptr) == 0,
							 "write particles/diameter");

		for (auto cellType : irange(0, cellTypes))
			copyElementParameter(cellType, colony.paramCells[cellType].elemType, uptr);
		    run_assert(gsd_write_chunk(&fp, "particles/typeid", GSD_TYPE_UINT32, totalElems, 1, 0, uptr) == 0,
							 "write particles/typeid");
	}

	if (SWITCH[OPT::CELLS_DIVIDING]) { // Status & Age (only useful for dividing particles)

		for (auto cellType : irange(0, cellTypes)) copyCellData(cellType, colony.cellTRho[cellType], fptr);
		run_assert(gsd_write_chunk(&fp, "particles/charge", GSD_TYPE_FLOAT, totalElems, 1, 0, fptr) == 0,
							 "write particles/charge");

//      std::function<float (const int &)> calcAge =
//      [&ts](const int &t0){return static_cast<float>((ts - t0)*paramsMD.dt);};
//      for(auto cellType : irange(0, cellTypes)){
//        if(colony.paramCells[cellType].DIV){
//          copyCellData(cellType, colony.cellT0[cellType], calcAge, fptr);
//        }else{
//          copyCellData(cellType, -1.0, fptr);
//        }
//      }
//      run_assert(gsd_write_chunk(&fp, "particles/mass", GSD_TYPE_FLOAT, totalElems, 1, 0, fptr) == 0,
//      "write particles/mass");
	}

	// particle positions
	for (auto cellType : irange(0, colony.getCellTypes())) {
		auto &params = colony.paramCells[cellType];
		array1<double> zhack(params.numElems);
		std::iota(zhack.begin(), zhack.end(), 0.0);
		for (auto &z : zhack) z *= -0.01 * units.Length;
		gsdBufferVectorShiftScale(cellType, colony.pos[cellType], hbox, zhack, fptr);
	}
	run_assert(gsd_write_chunk(&fp, "particles/position", GSD_TYPE_FLOAT, totalElems, 3, 0, fptr) == 0,
						 "write particles/position");

	// particle velocities
	for(auto cellType : irange(0, colony.getCellTypes())){
		gsdBufferVector(cellType, colony.vel[cellType], fptr);
		//else gsdBufferVectorScaleElem(cellType, colony.frc[cellType], colony.paramCells[cellType].zetaInv, fptr);
		//if(SWITCH[OPT::INTEGRATOR_DPD])
	}
	run_assert(gsd_write_chunk(&fp, "particles/velocity", GSD_TYPE_FLOAT, totalElems, 3, 0, fptr) == 0,
						 "write particles/velocity");

	// particle orientation
	//todo 从角度换算成四元数
////
//    for (auto cellType : irange(0, colony.getCellTypes())) gsdBufferQuaternion(cellType,colony.orien[cellType],fptr);
//    run_assert(gsd_write_chunk(&fp, "particles/orientation", GSD_TYPE_FLOAT, totalElems, 4, 0, fptr) == 0,
//               "write particles/orientation");

	// particle pressures
	for (auto cellType : irange(0, cellTypes)) copyCellData(cellType, colony.pressure[cellType], fptr);
	run_assert(gsd_write_chunk(&fp, "particles/mass", GSD_TYPE_FLOAT, totalElems, 1, 0, fptr) == 0,
						 "write particles/mass");
}

inline void initGSDOutput(gsd_handle &fp, const char *fileName) {
	run_assert(gsd_create(fileName, "CIL", "hoomd", gsd_make_version(1, 1)) == 0);
	run_assert(gsd_open(&fp, fileName, GSD_OPEN_APPEND) == 0);
	if (DIM == 2) {
		gsdBufferVector = gsdBufferVector2D;
		gsdBufferVectorScaleElem = gsdBufferVectorScaleElem2D;
		gsdBufferVectorShiftScale = gsdBufferVectorShiftScale2D;
		gsdBufferQuaternion = gsdBufferQuaternion2D;
	} else if (DIM == 3) {
		gsdBufferVector = gsdBufferVector3D;
		gsdBufferVectorScaleElem = gsdBufferVectorScaleElem3D;
		gsdBufferVectorShiftScale = gsdBufferVectorShiftScale3D;
		gsdBufferQuaternion = gsdBufferQuaternion3D;
	} else {
		run_assert(false, "Runtime Error : GSD DIM = 2|3");
	}
}

inline void outputGSDFrame(gsd_handle &fp, int id, int ts) {
	outputGSDConf(fp, ts);
	outputGSDParticles(fp, ts);
	run_assert(gsd_end_frame(&fp) == 0);
}

void outputLAMMPSFrame(std::ofstream &fp, int id, int ts) {
	using std::endl;
	snprintf(buffer, BUFFER_SIZE, "%s/LAMMPS/%s_%06d.lammps", paramsOut.dirName.c_str(), paramsOut.prjName.c_str(), ts);
	fp.open(buffer);
	fp << "ITEM: TIMESTEP\n";
	fp << id << endl;
	fp << "ITEM: NUMBER OF ATOMS\n";
	fp << format("%ld\n") % colony.getTotalElems();
	fp << "ITEM: BOX BOUNDS pp pp pp\n";
	fp << format("0 %lf\n 0 %lf\n 0 %lf \n") % container.box.lx() % container.box.ly() % 1.0;
	{
		fp << "ITEM: ATOMS id type x y z fx fy fz radius Age isMotile\n";
		format fmtElem("%d %d %12.6f %12.6f %7.4f %12.6f %12.6f 0.0 %12.6f %12.6f %d\n");
		for (auto cellType : irange(0, colony.getCellTypes())) {
			auto &params = colony.paramCells[cellType];
			auto &pos = colony.pos[cellType];
			auto &frc = colony.frc[cellType];
			auto &sig = colony.sigmaCore[cellType];
			auto &state = colony.cellState[cellType];
			auto &tBirth = colony.cellT0[cellType];
			auto &cumul = colony.cumulElems[cellType];
			auto nCells = colony.getNumCells(cellType);

			// z-depth hack
			vector<double> zhack(params.numElems);
			std::iota(zhack.begin(), zhack.end(), 0.0);
			for (auto &z : zhack) z *= -0.01 * units.Length;

			// get cell age
			array1<float> cellAge(nCells, colony.getVectorBuffer());
			if (params.DIV) {
				for (auto i : irange(0, nCells)) cellAge[i] = ((ts - tBirth[i]) * paramsMD.dt);
			} else {
				std::fill_n(cellAge(), cellAge.Size(), -1.0);
			}

			for (auto i : irange(0, nCells)) {
				const auto &Ri = pos[i];
				const auto &Fi = frc[i];
				const auto &Si = sig[i];
				for (auto j : irange(0, params.numElems))
					fp << fmtElem % (cumul + i * params.numElems + j) % params.elemType(j) % Ri(j, 0) % Ri(j, 1) %
								zhack[j]
								% Fi(j, 0) % Fi(j, 1) % (Si[j] / 2) % cellAge[i] % state[i];
			}
		}
	}
	fp.close();
}


void outputFrame(int id, int ts) {
	if (SWITCH[OPT::OUTPUT_H5]) outputHDF5Frame(h5Out, id, ts);
	if (SWITCH[OPT::OUTPUT_GSD]) outputGSDFrame(gsdOut, id, ts);
	if (SWITCH[OPT::OUTPUT_LAMMPS]) outputLAMMPSFrame(lammpsOut, id, ts);
}

void initializeOutput() {
	lz2d = colony.getMaxSigma();
	if (SWITCH[OPT::OUTPUT_DUMP])
		run_assert(dirCheckMake(paramsOut.dirName.c_str()), "Error creating output directory");
	if (SWITCH[OPT::OUTPUT_H5]) initHDF5Output(h5Out, (paramsOut.dirName + "/" + paramsOut.prjName + ".h5").c_str());
	if (SWITCH[OPT::OUTPUT_GSD]) initGSDOutput(gsdOut, (paramsOut.dirName + "/" + paramsOut.prjName + ".gsd").c_str());
	if (SWITCH[OPT::OUTPUT_LAMMPS])
		run_assert(dirCheckMake((paramsOut.dirName + string("/LAMMPS")).c_str()), "Error creating lammps directory");
}

void finalizeOutput() {
	if (SWITCH[OPT::OUTPUT_H5]) h5Out.close_file();
	if (SWITCH[OPT::OUTPUT_GSD]) gsd_close(&gsdOut);
	{
		snprintf(buffer, BUFFER_SIZE, "%s.Last.gsd", paramsOut.prjName.c_str());
		initGSDOutput(gsdOut, buffer);
		outputGSDFrame(gsdOut, 0, 0);
		gsd_close(&gsdOut);
		std::string sPath = paramsOut.dirName + "/Last/";
		mode_t nMode = 0733; // UNIX style permissions
		mkdir(sPath.c_str(), nMode); // can be used on non-Windows
		auto newname = (paramsOut.dirName + "/Last/" + paramsOut.prjName + ".Last.gsd").c_str();
		std::rename(buffer, newname);
	}
	{
		snprintf(buffer, BUFFER_SIZE, "%s.Last.h5", paramsOut.prjName.c_str());
		h5Out.create_file(buffer);
		outputHDF5Frame(h5Out, -1, -1);
		h5Out.close_file();
		snprintf(buffer, BUFFER_SIZE, "%s.Last.h5", paramsOut.prjName.c_str());
		auto newname = (paramsOut.dirName + "/Last/" + paramsOut.prjName + ".Last.h5").c_str();
		std::rename(buffer, newname);
	}
}
