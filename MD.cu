#include "MD.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern double potentialEnergy;

__global__ void initLJParamsKernel(double* epsilon, double* rcut2, const ElemParams* params, double* rc2, int elemTypes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elemTypes) {
        double eps_i = params[i].epsilonWell;
        for (int j = i; j < elemTypes; j++) {
            double eps_j = params[j].epsilonWell;
            epsilon[i * elemTypes + j] = epsilon[j * elemTypes + i] = 24.0 * sqrt(eps_i * eps_j);
            rcut2[i * elemTypes + j] = rcut2[j * elemTypes + i] = rc2[i * elemTypes + j];
        }
    }
}

void LJParams::init(const vector<ElemParams> &params, const array2<double> &rc2) {
    int elemTypes = params.size();
    epsilon.Allocate(elemTypes, elemTypes);
    epsilon = 0.0;
    rcut2.Allocate(elemTypes, elemTypes);
    rcut2 = 0.0;

    // Allocate memory on device
    vector<ElemParams> *d_params;
    array2<double> *d_rc2;
    array2<double> *d_epsilon;
    array2<double> *d_rcut2;

    cudaMalloc(&d_params, sizeof(ElemParams) * params.size());
    cudaMalloc(&d_rc2, sizeof(double) * rc2.Size());
    cudaMalloc(&d_epsilon, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_rcut2, sizeof(double) * elemTypes * elemTypes);

    // Copy data to device
    cudaMemcpy(d_params, params, sizeof(ElemParams) * params.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc2, rc2, sizeof(double) * rc2.Size(), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (elemTypes + threadsPerBlock - 1) / threadsPerBlock;
    initLJParamsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_epsilon, d_rcut2, d_params, d_rc2, elemTypes);

    // Copy results back to host
    cudaMemcpy(epsilon, d_epsilon, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(rcut2, d_rcut2, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_params);
    cudaFree(d_rc2);
    cudaFree(d_epsilon);
    cudaFree(d_rcut2);
}

__global__ void initSCParamsKernel(double* epsilonCore, double* epsilonWell, double* sigmaWell, double* rcut2, const ElemParams* params, double* rc2, int elemTypes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elemTypes) {
        double eps0_i = params[i].epsilonCore;
        double eps1_i = params[i].epsilonWell;
        double sig1_i = params[i].sigmaWell;
        for (int j = i; j < elemTypes; j++) {
            double eps0_j = params[j].epsilonCore;
            double eps1_j = params[j].epsilonWell;
            double sig1_j = params[j].sigmaWell;

            double eps0_ij = sqrt(eps0_i * eps0_j);
            double eps1_ij = sqrt(eps1_i * eps1_j);

            epsilonCore[i * elemTypes + j] = epsilonCore[j * elemTypes + i] = eps0_ij;
            epsilonWell[i * elemTypes + j] = epsilonWell[j * elemTypes + i] = eps1_ij;
            sigmaWell[i * elemTypes + j] = sigmaWell[j * elemTypes + i] = 0.5 * (sig1_i + sig1_j);
            rcut2[i * elemTypes + j] = rcut2[j * elemTypes + i] = rc2[i * elemTypes + j];
        }
    }
}

void SCParams::init(const vector<ElemParams> &params, const array2<double> &rc2) {
    int elemTypes = params.size();
    epsilonCore.Allocate(elemTypes, elemTypes);
    epsilonWell.Allocate(elemTypes, elemTypes);
    sigmaWell.Allocate(elemTypes, elemTypes);
    rcut2.Allocate(elemTypes, elemTypes);

    // Allocate memory on device
    vector<ElemParams> *d_params;
    array<double> *d_rc2, *d_epsilonCore, *d_epsilonWell, *d_sigmaWell, *d_rcut2;
    cudaMalloc(&d_params, sizeof(ElemParams) * params.size());
    cudaMalloc(&d_rc2, sizeof(double) * rc2.Size());
    cudaMalloc(&d_epsilonCore, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_epsilonWell, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_sigmaWell, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_rcut2, sizeof(double) * elemTypes * elemTypes);

    // Copy data to device
    cudaMemcpy(d_params, params, sizeof(ElemParams) * params.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc2, rc2, sizeof(double) * rc2.Size(), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (elemTypes + threadsPerBlock - 1) / threadsPerBlock;
    initSCParamsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_epsilonCore, d_epsilonWell, d_sigmaWell, d_rcut2, d_params, d_rc2, elemTypes);

    // Copy results back to host
    cudaMemcpy(epsilonCore.data(), d_epsilonCore, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(epsilonWell.data(), d_epsilonWell, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sigmaWell.data(), d_sigmaWell, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(rcut2.data(), d_rcut2, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_params);
    cudaFree(d_rc2);
    cudaFree(d_epsilonCore);
    cudaFree(d_epsilonWell);
    cudaFree(d_sigmaWell);
    cudaFree(d_rcut2);
}


__global__ void initHZParamsKernel(double* epsilonCore, double* epsilonWell, double* sigmaWell, double* rcut2, const ElemParams* params, double* rc2, int elemTypes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elemTypes) {
        double eps0_i = params[i].epsilonCore;
        double eps1_i = params[i].epsilonWell;
        double sig1_i = params[i].sigmaWell;
        for (int j = i; j < elemTypes; j++) {
            double eps0_j = params[j].epsilonCore;
            double eps1_j = params[j].epsilonWell;
            double sig1_j = params[j].sigmaWell;

            double eps0_ij = sqrt(eps0_i * eps0_j);
            double eps1_ij = sqrt(eps1_i * eps1_j);

            epsilonCore[i * elemTypes + j] = epsilonCore[j * elemTypes + i] = eps0_ij;
            epsilonWell[i * elemTypes + j] = epsilonWell[j * elemTypes + i] = eps1_ij;
            sigmaWell[i * elemTypes + j] = sigmaWell[j * elemTypes + i] = 0.5 * (sig1_i + sig1_j);
            rcut2[i * elemTypes + j] = rcut2[j * elemTypes + i] = rc2[i * elemTypes + j];
        }
    }
}

void HZParams::init(const vector<ElemParams> &params, const array2<double> &rc2) {
    int elemTypes = params.size();
    epsilonCore.Allocate(elemTypes, elemTypes);
    epsilonWell.Allocate(elemTypes, elemTypes);
    sigmaWell.Allocate(elemTypes, elemTypes);
    rcut2.Allocate(elemTypes, elemTypes);

    // Allocate memory on device
    vector<ElemParams> *d_params;
    array2<double> *d_rc2, *d_epsilonCore, *d_epsilonWell, *d_sigmaWell, *d_rcut2;
    cudaMalloc(&d_params, sizeof(ElemParams) * params.size());
    cudaMalloc(&d_rc2, sizeof(double) * rc2.Size());
    cudaMalloc(&d_epsilonCore, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_epsilonWell, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_sigmaWell, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_rcut2, sizeof(double) * elemTypes * elemTypes);

    // Copy data to device
    cudaMemcpy(d_params, params.data(), sizeof(ElemParams) * params.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc2, rc2.data(), sizeof(double) * rc2.Size(), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (elemTypes + threadsPerBlock - 1) / threadsPerBlock;
    initHZParamsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_epsilonCore, d_epsilonWell, d_sigmaWell, d_rcut2, d_params, d_rc2, elemTypes);

    // Copy results back to host
    cudaMemcpy(epsilonCore.data(), d_epsilonCore, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(epsilonWell.data(), d_epsilonWell, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sigmaWell.data(), d_sigmaWell, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(rcut2.data(), d_rcut2, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_params);
    cudaFree(d_rc2);
    cudaFree(d_epsilonCore);
    cudaFree(d_epsilonWell);
    cudaFree(d_sigmaWell);
    cudaFree(d_rcut2);
}

__global__ void initEAParamsKernel(double* epsilonCore, double* epsilonWell, double* sigmaWell, double* rcut2, const ElemParams* params, double* rc2, int elemTypes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elemTypes) {
        double eps0_i = params[i].epsilonCore;
        double eps1_i = params[i].epsilonWell;
        double sig1_i = params[i].sigmaWell;
        for (int j = i; j < elemTypes; j++) {
            double eps0_j = params[j].epsilonCore;
            double eps1_j = params[j].epsilonWell;
            double sig1_j = params[j].sigmaWell;

            double eps0_ij = sqrt(eps0_i * eps0_j);
            double eps1_ij = sqrt(eps1_i * eps1_j);

            epsilonCore[i * elemTypes + j] = epsilonCore[j * elemTypes + i] = eps0_ij;
            epsilonWell[i * elemTypes + j] = epsilonWell[j * elemTypes + i] = eps1_ij;
            sigmaWell[i * elemTypes + j] = sigmaWell[j * elemTypes + i] = 0.5 * (sig1_i + sig1_j);
            rcut2[i * elemTypes + j] = rcut2[j * elemTypes + i] = rc2[i * elemTypes + j];
        }
    }
}

void EAParams::init(const vector<ElemParams> &params, const array2<double> &rc2) {
    int elemTypes = params.size();
    epsilonCore.Allocate(elemTypes, elemTypes);
    epsilonWell.Allocate(elemTypes, elemTypes);
    sigmaWell.Allocate(elemTypes, elemTypes);
    rcut2.Allocate(elemTypes, elemTypes);

    // Allocate memory on device
    vector<ElemParams> *d_params;
    array2<double> *d_rc2, *d_epsilonCore, *d_epsilonWell, *d_sigmaWell, *d_rcut2;
    cudaMalloc(&d_params, sizeof(ElemParams) * params.size());
    cudaMalloc(&d_rc2, sizeof(double) * rc2.Size());
    cudaMalloc(&d_epsilonCore, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_epsilonWell, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_sigmaWell, sizeof(double) * elemTypes * elemTypes);
    cudaMalloc(&d_rcut2, sizeof(double) * elemTypes * elemTypes);

    // Copy data to device
    cudaMemcpy(d_params, params.data(), sizeof(ElemParams) * params.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc2, rc2.data(), sizeof(double) * rc2.Size(), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (elemTypes + threadsPerBlock - 1) / threadsPerBlock;
    initEAParamsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_epsilonCore, d_epsilonWell, d_sigmaWell, d_rcut2, d_params, d_rc2, elemTypes);

    // Copy results back to host
    cudaMemcpy(epsilonCore.data(), d_epsilonCore, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(epsilonWell.data(), d_epsilonWell, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sigmaWell.data(), d_sigmaWell, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);
    cudaMemcpy(rcut2.data(), d_rcut2, sizeof(double) * elemTypes * elemTypes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_params);
    cudaFree(d_rc2);
    cudaFree(d_epsilonCore);
    cudaFree(d_epsilonWell);
    cudaFree(d_sigmaWell);
    cudaFree(d_rcut2);
}

/*!
  Compute Fene force
 */
template<typename T>
inline __host__ __device__ T d_SQ(const T &a) {
    return a * a;
}

inline __host__ __device__ double FENE(const double &kappa, const double &r2max, const double &r2) {
	return -kappa / (1.0 - d_SQ(r2 / r2max));
}

inline __host__ __device__ double FENE_Potential(const double &kappa, const double &r2max, const double &r2) {
	return -0.5 * kappa * r2max * log(1 - (r2 / r2max));
}

inline __host__ __device__ double Spring(const double &kappa_push, const double &r2max, const double &r2, const double &spring_ratio) {
	return kappa_push * fabs(r2 - (r2max * spring_ratio));
}

// maximum
template<typename T>
inline __device__ T d_MAX(const T &a, const T &b) {
    return (a >= b ? a : b);
}

inline __device__ double d_twoSigma2ContactArea(const double &r, const double &a, const double &b) {
    auto ra = a / 2;
    auto rb = b / 2;
    auto tempA = sqrt(4 * d_SQ(r) * d_SQ(ra) - d_SQ(d_SQ(r) - d_SQ(rb) + d_SQ(ra))) / (2 * r);
    return (tempA > 0 ? Constants::PI * d_SQ(tempA) : 0);
}

inline __device__ double d_twoSigma2ContactLength(const double &a, const double &b, const double &h) {
	auto ra = a / 2;
	auto rb = b / 2;
	auto tempA = 2*sqrt(d_SQ(ra)-d_SQ(ra-(2*rb*h-d_SQ(h))/(2*(ra+rb-h))));
	auto small_r = (a<=b) ? a:b;
	tempA = isnan(tempA)? small_r:tempA;
	return tempA;
}

__global__ void initRandomGenerator(curandState *state, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ double generateRandomDouble(curandState *state, double x) {
    return curand_uniform_double(state) * x;
}


/*!
Calculates the forces between two cells as well as the pressures arising from those forces acting on both cells.
*/
__device__ void cellCellForcePressureDevice(const CellParams &paramsi, const CellParams &paramsj, 
    const cellIDs &ci, const cellIDs &cj, 
    const array2<double> &ri, const array2<double> &rj, 
    array2<double> &fi, array2<double> &fj, 
    double &pri, double &prj, 
    const ParamCells *paramCells, const ParamElems *paramElems, 
    const double *sigmaCore, const Container &container, curandState *state) {
double f0, phi;
double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double nij[MAX_DIM] = {0.0, 0.0, 0.0};
for (int iElemId = 0; iElemId < paramCells[ci.type].numElems; iElemId++) {
int iElemType = paramCells[ci.type].elemType[iElemId];
ParamElems &pi = paramElems[iElemType];
pi.updateCore(sigmaCore[ci.type * numElems + ci.id * numElems + iElemId]);
double sigmai = sigmaCore[ci.type * numElems + ci.id * numElems + iElemId];
for (int jElemId = 0; jElemId < paramCells[cj.type].numElems; jElemId++) {
int jElemType = paramCells[cj.type].elemType[jElemId];
ParamElems &pj = paramElems[jElemType];
pj.updateCore(sigmaCore[cj.type * numElems + cj.id * numElems + jElemId]);
double sigmaj = sigmaCore[cj.type * numElems + cj.id * numElems + jElemId];
double r = sqrt(container.box.distance(rij, ri[iElemId], rj[jElemId]));
if (r <= (sigmai + sigmaj) / 2) {
for (int d = 0; d < MAX_DIM; d++) {
nij[d] = rij[d] / r;
}
if (MAX_DIM == 2 && rij[0] == 0 && rij[1] == 0){
// 创建 curand 状态对象
int idx = threadIdx.x + blockIdx.x * blockDim.x;
curandState state;
curand_init(static_cast<unsigned long long>(clock()) + idx, idx, 0, &state);
//double alpha = Random::uniform0x(Constants::TWOPI);
double alpha = generateRandomDouble(state, Constant::TWOPI);
double x = sin(alpha);
double y = cos(alpha);
rij[0] = x;
rij[1] = y;
}
double h;
if (SWITCH[OPT::FORCE_HERTZIAN] || SWITCH[OPT::FORCE_EA]) h = d_MAX(((sigmai + sigmaj) / 2) - r, 0.0);
else h = r;
if (paramsForce->elemForce(pi, pj, h, f0, sigmai, sigmaj)) {
double Aij = (MAX_DIM == 3 ? d_twoSigma2ContactArea(r, sigmai, sigmaj) : d_twoSigma2ContactLength(sigmai, sigmaj, h));
f0 += pi.epsilonWell * Aij;
double temp = (Aij == 0 ? 0 : (4 * Constants::PI * d_SQ(sigmai / 2) / 3));
temp = (MAX_DIM == 3 ? temp : Aij);
if (f0 < 0.0 || ci.type == cj.type){
pri -= f0 / temp;
prj -= f0 / temp;
for (int d = 0; d < MAX_DIM; d++) {
atomicAdd(&fi[iElemId][d], f0 * rij[d]);
atomicAdd(&fj[jElemId][d], -f0 * rij[d]);
}
} else {
pri -= paramsi.crossAdhesion * f0 / temp;
prj -= paramsj.crossAdhesion * f0 / temp;
for (int d = 0; d < MAX_DIM; d++) {
atomicAdd(&fi[iElemId][d], paramsi.crossAdhesion * f0 * rij[d]);
atomicAdd(&fj[jElemId][d], -paramsj.crossAdhesion * f0 * rij[d]);
}
}
}
}
}
}
}

__global__ void cellInteractionsKernel(const CellParams *params, const cellIDs *cellInfos, 
    const array2<double> *positions, array2<double> *forces, 
    array2<double> *forces2, array2<double> *forces3, 
    const array2<double> *velocities, double *pressures, 
    const ParamCells *paramCells, const ParamElems *paramElems, 
    const double *sigmaCore, const Container *container, int nCells, curandState *state,) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
curandState localState = state[idx];//@@@@@@@初始化随机数
if (idx < nCells) {
cellIDs ci = cellInfos[idx];
array2<double> ri = positions[ci.id];
array2<double> fi = forces[ci.id];
array2<double> fid = forces2[ci.id];
array2<double> fir = forces3[ci.id];
array2<double> vi = velocities[ci.id];
double pri = pressures[ci.id];
const CellParams &paramsi = params[ci.type];

for (int j = idx + 1; j < nCells; j++) {
cellIDs cj = cellInfos[j];
array2<double> rj = positions[cj.id];
array2<double> fj = forces[cj.id];
array2<double> fjd = forces2[cj.id];
array2<double> fjr = forces3[cj.id];
array2<double> vj = velocities[cj.id];
double prj = pressures[cj.id];
const CellParams &paramsj = params[cj.type];

cellCellForcePressureDevice(paramsi, paramsj, ci, cj, ri, rj, fi, fj, pri, prj, paramCells, paramElems, sigmaCore, *container, &localState);
state[idx] = localState;//@@@@@@@@@更新随机数
}
pressures[ci.id] = pri;
}
}

void calculateCellInteractions(const CellParams *params, const cellIDs *cellInfos, 
    const array2<double> *positions, array2<double> *forces, 
    array2<double> *forces2, array2<double> *forces3, 
    const array2<double> *velocities, double *pressures, 
    const ParamCells *paramCells, const ParamElems *paramElems, 
    const double *sigmaCore, const Container *container, int nCells) {
// 分配设备内存
CellParams *d_params;
cellIDs *d_cellInfos;
array2<double> *d_positions, *d_forces, *d_forces2, *d_forces3, *d_velocities;
double *d_pressures;
ParamCells *d_paramCells;
ParamElems *d_paramElems;
double *d_sigmaCore;
Container *d_container;

cudaMalloc((void**)&d_params, sizeof(CellParams) * nCells);
cudaMalloc((void**)&d_cellInfos, sizeof(cellIDs) * nCells);
cudaMalloc((void**)&d_positions, sizeof(array2<double>) * nCells);
cudaMalloc((void**)&d_forces, sizeof(array2<double>) * nCells);
cudaMalloc((void**)&d_forces2, sizeof(array2<double>) * nCells);
cudaMalloc((void**)&d_forces3, sizeof(array2<double>) * nCells);
cudaMalloc((void**)&d_velocities, sizeof(array2<double>) * nCells);
cudaMalloc((void**)&d_pressures, sizeof(double) * nCells);
cudaMalloc((void**)&d_paramCells, sizeof(ParamCells) * nCells);
cudaMalloc((void**)&d_paramElems, sizeof(ParamElems) * nCells);
cudaMalloc((void**)&d_sigmaCore, sizeof(double) * nCells * numElems);
cudaMalloc((void**)&d_container, sizeof(Container));

// 复制数据到设备
cudaMemcpy(d_params, params, sizeof(CellParams) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_cellInfos, cellInfos, sizeof(cellIDs) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_positions, positions, sizeof(array2<double>) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_forces, forces, sizeof(array2<double>) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_forces2, forces2, sizeof(array2<double>) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_forces3, forces3, sizeof(array2<double>) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_velocities, velocities, sizeof(array2<double>) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_pressures, pressures, sizeof(double) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_paramCells, paramCells, sizeof(ParamCells) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_paramElems, paramElems, sizeof(ParamElems) * nCells, cudaMemcpyHostToDevice);
cudaMemcpy(d_sigmaCore, sigmaCore, sizeof(double) * nCells * numElems, cudaMemcpyHostToDevice);
cudaMemcpy(d_container, container, sizeof(Container), cudaMemcpyHostToDevice);

// 计算线程和块的数量
int threadsPerBlock = 256;
int blocksPerGrid = (nCells + threadsPerBlock - 1) / threadsPerBlock;

//随机数
curandState *d_states;
cudaMalloc(&d_states, threadsPerBlock * blocksPerGrid * sizeof(curandState));
initRandomGenerator<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL));
// 调用内核函数
cellInteractionsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_params, d_cellInfos, d_positions, d_forces, d_forces2, d_forces3, d_velocities, d_pressures, d_paramCells, d_paramElems, d_sigmaCore, d_container, nCells, d_states);

// 将结果复制回主机
cudaMemcpy(pressures, d_pressures, sizeof(double) * nCells, cudaMemcpyDeviceToHost);

// 释放设备内存
cudaFree(d_params);
cudaFree(d_cellInfos);
cudaFree(d_positions);
cudaFree(d_forces);
cudaFree(d_forces2);
cudaFree(d_forces3);
cudaFree(d_velocities);
cudaFree(d_pressures);
cudaFree(d_paramCells);
cudaFree(d_paramElems);
cudaFree(d_sigmaCore);
cudaFree(d_container);
}

void calculInterForces(const vector<array2<double>> &positions, const vector<array2<double>> &forces,
    const vector<array2<double>> &forces2, const vector<array2<double>> &forces3,
    const vector<array2<double>> &velocities, vector<array1<double>> &pressures) {
    if (SWITCH[OPT::FORCE_GHOSTS]) return;

    int nCells = colony.getTotalCells();

    // 创建临时数组来传递给 CUDA 函数
    CellParams *params = new CellParams[nCells];
    cellIDs *cellInfos = new cellIDs[nCells];
    array2<double> *positionsArray = new array2<double>[nCells];
    array2<double> *forcesArray = new array2<double>[nCells];
    array2<double> *forces2Array = new array2<double>[nCells];
    array2<double> *forces3Array = new array2<double>[nCells];
    array2<double> *velocitiesArray = new array2<double>[nCells];
    double *pressuresArray = new double[nCells];

    // 填充临时数组
    for (int i = 0; i < nCells; i++) {
    params[i] = colony.paramCells[i];
    cellInfos[i] = colony.UID2Cell(i);
    positionsArray[i] = positions[i];
    forcesArray[i] = forces[i];
    forces2Array[i] = forces2[i];
    forces3Array[i] = forces3[i];
    velocitiesArray[i] = velocities[i];
    pressuresArray[i] = pressures[i];
    }

    // 调用 CUDA 函数
    calculateCellInteractions(params, cellInfos, positionsArray, forcesArray, forces2Array, forces3Array, velocitiesArray, pressuresArray, colony.paramCells.data(), colony.paramElems.data(), colony.sigmaCore.data(), &colony.container, nCells);

    // 将结果复制回原始数组
    for (int i = 0; i < nCells; i++) {
    pressures[i] = pressuresArray[i];
    }

    // 释放临时数组
    delete[] params;
    delete[] cellInfos;
    delete[] positionsArray;
    delete[] forcesArray;
    delete[] forces2Array;
    delete[] forces3Array;
    delete[] velocitiesArray;
    delete[] pressuresArray;
}



void binaryCellForce(const CellParams &params, const int &state, const array2<double> &elemPos,
    const array2<double> &elemFrc, const array2<double> &elemVel, const array1<double> &Wiener,
    const array1<double> &sig) {
assert(params.numElems == 2 && elemPos.Nx() == 2 && elemFrc.Nx() == 2); // check element number
assert(elemPos.Ny() == DIM && elemFrc.Ny() == DIM); // check dimensions
// vector from back (0) to front (1) elements
double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double r2 = container.box.distance(rij, elemPos[0], elemPos[1]); // rij = r1 - r0
double r = sqrt(r2);
if (r2 == 0.0) { // add small random displacement in case of overlap
r = Constants::BIG_TOL_MP * 0.5 * (params.sigmaCore[0] + params.sigmaCore[1]);
Random::onDSphere(DIM, rij);
for (auto d : irange(0, DIM)) rij[d] *= r;
r2 = SQ(r);
container.box.updatePos(elemPos[1], rij);
}

double activity[2] = {params.m[0], params.m[1]};
double nij[MAX_DIM] = {rij[0], rij[1], rij[2]};
double strongContract = 2;

// FENE potential between back (0) and front (1) elements
double r2max0 = (state == CellState::Dividing ? params.divisionR2max(0, 1) : params.r2max(0, 1));
double r2max = r2max0 > sig.Max() ? r2max0 : sig.Max();
assert(positive_mp(r2max - r2));

double fene = FENE(params.kappa(0, 1), r2max, r2);
double spring;
double springRatio = params.spring_ratio(0, 1);
spring =
r2 <= (r2max * springRatio) ? Spring(params.kappa_push(0, 1), r2max, r2, springRatio) : FENE(params.kappa(0, 1),
                                                                                                                                                                r2max *
                                                                                                                                                                (1 - springRatio),
                                                                                                                                                                r2 -
                                                                                                                                                                r2max * springRatio);
double feneDivMax = FENE(params.kappa(0, 1), r2max, params.divisionR2min);  //the max fene during division

double k = (FENE(params.kappa(0, 1), r2max, 0.98) - FENE(params.kappa(0, 1), r2max, 0.95)) / (0.08);
// If Dividing apply opposite motility force to both elements
if (state == CellState::Dividing) {
for (auto i = 0; i <= 1; i++) {
activity[i] = (-params.growthk0[i] * feneDivMax + params.growthk1[i] + params.growthk2[i] * SQ(r2));
}
strongContract = 1;
} else if (!params.CIL) { // not dividing & no contact inhibition
double scaling = params.rss(0, 1) / r;
for (auto d : irange(0, DIM)) nij[d] *= scaling;
}
double rndForce[MAX_DIM] = {0.0, 0.0, 0.0};
if (SWITCH[OPT::NOISE_ACTIVE]) { // DIMENSION WARNING
double Wr = Wiener[0];
double Wt = Wiener[1];
rndForce[0] = 0.5 * (Wr * rij[0] / r - Wt * rij[1]);
rndForce[1] = 0.5 * (Wr * rij[1] / r + Wt * rij[0]);
}

array1<double>::opt vel0 = elemVel[0];
array1<double>::opt vel1 = elemVel[1];
array1<double>::opt frc0 = elemFrc[0];
array1<double>::opt frc1 = elemFrc[1];
for (auto d : irange(0, DIM)) {
// Take the drag friction and intra-cellular friction into consideration
// forces on back element

auto tempCon0 = -(activity[0] * nij[d] + spring * rij[d] * strongContract + rndForce[d] * rij[d]);
auto tempCon1 = (activity[1] * nij[d] + spring * rij[d] * strongContract + rndForce[d] * rij[d]);

auto tempFriction0 = params.zeta[0] * SQ(vel0[d]) * (vel0[d] / ABS(vel0[d]));
auto tempFriction1 = params.zeta[1] * SQ(vel1[d]) * (vel1[d] / ABS(vel1[d]));

frc0[d] = ((std::isinf(tempFriction0) || std::isnan(tempFriction0)) ? tempCon0 : (tempCon0 - tempFriction0));
frc1[d] = ((std::isinf(tempFriction1) || std::isnan(tempFriction1)) ? tempCon1 : (tempCon1 - tempFriction1));
}
}

void singleCellForce(const CellParams &params, const int &state, const array2<double> &elemPos,
    const array2<double> &elemFrc, const array2<double> &elemVel, const array1<double> &Wiener,
    const array1<double> &sig) {

array1<double>::opt vel0 = elemVel[0];
array1<double>::opt frc0 = elemFrc[0];

// add small random velocity in case velocity is zero so that the DPD friction term is always defined.
double velLength = vel0.L2();
if(velLength == 0.0){
Random::onDSphere(DIM, vel0);
for (auto d : irange(0, DIM)) {
vel0[d] *= Constants::BIG_TOL_MP;
}
}
velLength = vel0.L2();

// Migration force for single cell force:
// - Currently: make motility point in direction of velocity
// - TODO Better: have a persistent motility that relaxes toward the velocity.
// - TODO vector and dynamics describing the polarity of the cell; rename rij, nij
double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double activity = params.m[0];

// Use velocity as a proxy for cell polarity
double nij[MAX_DIM] = {rij[0], rij[1], rij[2]};
for (auto d : irange(0, DIM)) {
nij[d] = vel0[d] / velLength;
}

double rndForce[MAX_DIM] = {0.0, 0.0, 0.0};
if (SWITCH[OPT::NOISE_ACTIVE]) { // DIMENSION WARNING
std::cout << "Warning, active noise doesn't currently do anything for a single particle cell" << std::endl; // Put this warning into Input.cpp
}

for (auto d : irange(0, DIM)) {
// Take the drag friction and intra-cellular friction into consideration
auto tempCon0 = activity * nij[d] + rndForce[d] * rij[d];
// auto tempFriction0 = params.zeta[0] * SQ(vel0[d]) * (vel0[d] / ABS(vel0[d]));
// auto tempFriction0 = (SWITCH[OPT::INTEGRATOR_DPD] ? tempFriction0 : 0);
// frc0[d] = ((std::isinf(tempFriction0) || std::isnan(tempFriction0)) ? tempCon0 : (tempCon0 - tempFriction0));
auto tempFriction0 = (SWITCH[OPT::INTEGRATOR_DPD] ? params.zeta[0] * SQ(vel0[d]) * (vel0[d] / ABS(vel0[d])) : 0);
frc0[d] = (tempCon0 - tempFriction0); // This should always work now that vel0 is never 0.
}
}

/*!
  Given position of one of the elements (master), reposition both elements randomly, keeping the center of mass equal to the master position
  SINGLE: random choose a direction to put two daughter cells inside the current volume of the mother cell
 */
inline void binaryRandomReorientation(const array1<double> &elemSlave, const array1<double> &elemMaster, const double &radius) {
	for (auto d : irange(0, DIM)) elemSlave[d] = elemMaster[d];
	double dr[MAX_DIM] = {0.0, 0.0, 0.0};
	Random::onDSphere(DIM, dr);
	container.box.updatePos(elemMaster, dr, radius);
	container.box.updatePos(elemSlave, dr, -radius);
}

inline void singleRandomReorientation(const array1<double> &elemSlave, const array1<double> &elemMaster, const array1<double> dr) {
	for (auto d : irange(0, DIM)) elemSlave[d] = elemMaster[d];
	//todo to compatiable with the single division process
	container.box.updatePos(elemMaster, dr);
	for (auto d : irange(0, DIM)) {
		dr[d] *= (-1);
	}
	container.box.updatePos(elemSlave, dr);
}

void singleCellDivision(const double &rr, const double &sigmaTheta, const double &sigmaCore, const array2<double> &motherPos,
    const array2<double> &daughterPos) {
//assert(params.numElems == 1 && motherPos.Nx() == 1 && daughterPos.Nx() == 1);
//assert(motherPos.Ny() == DIM && daughterPos.Ny() == DIM);

//let two new daughter cells keep the space left by the one mother cell
if (DIM == 2) {
// Total volume of the two daughter cells equal the volume of the single mother cell
auto lMom = sigmaCore;
auto rMom = sigmaCore / 2;
auto areaMom = Constants::PI * rMom * rMom;

double lDaughter = lMom / sqrt(2);
//run_assert(lDaughter > 0.50, "lDaughter too small\n");
//run_assert(lDaughter < 2 , " lDaughter too large\n");
auto r = (lMom - lDaughter) / 2;


auto alpha = Random::uniform0x(Constants::TWOPI);

auto x = r * sin(alpha);
auto y = r * cos(alpha);

double dr[2] = {x, y};
for (auto d : irange(0, DIM)) daughterPos(0, d) = motherPos(0, d);
singleRandomReorientation(motherPos[0], daughterPos[0], dr);
} else if (DIM == 3) {
// Total volume of the two daughter cells equal the volume of the single mother cell
auto lMom = sigmaCore;
auto rMom = sigmaCore / 2;
auto volumeMom = 4 * Constants::PI * rMom * rMom * rMom / 3;

double a = 9 * lMom / 4;
double b = -(3 * (lMom * lMom) / 2);
double c = lMom * lMom * lMom / 4 - 3 * volumeMom / (Constants::PI);
double lDaughter = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
//run_assert(lDaughter > 0.50, "lDaughter too small\n");
//run_assert(lDaughter < 2 , " lDaughter too large\n");
auto r = (lMom - lDaughter) / 2;

auto alpha = Random::uniform0x(Constants::TWOPI);
auto beta = Random::uniform0x(Constants::TWOPI);

auto x = r * sin(alpha) * cos(beta);
auto y = r * sin(alpha) * sin(beta);
auto z = r * cos(alpha);

double dr[3] = {x, y, z};
for (auto d : irange(0, DIM)) daughterPos(0, d) = motherPos(0, d);
singleRandomReorientation(motherPos[0], daughterPos[0], dr);
}
}

void binaryCellDivision(const double &rr, const double &sigmaTheta, const double &sigmaCore, const array2<double> &motherPos,
    const array2<double> &daughterPos) {
//assert(params.numElems == 2 && motherPos.Nx() == 2 && daughterPos.Nx() == 2);
//assert(motherPos.Ny() == DIM && daughterPos.Ny() == DIM);

// Mother keeps front cell - Daughter inherits back cell
for (auto d : irange(0, DIM)) daughterPos(0, d) = motherPos(0, d);

binaryRandomReorientation(motherPos[0], motherPos[1], sigmaCore);      // Mother keeps front cell
binaryRandomReorientation(daughterPos[1], daughterPos[0], sigmaCore);      // Daughter inherits back cell
}

/*!
Reposition cell elements after failed cell division event, to prevent unphysical acceleration caused by stretching during division
*/
void binaryCellDivisionFail(const CellParams &params, const array2<double> &motherPos) {
assert(params.numElems == 2 && motherPos.Nx() == 2);
assert(motherPos.Ny() == DIM);

// vector from back to front
double dr[MAX_DIM] = {0.0, 0.0, 0.0};
container.box.distance(dr, motherPos[0], motherPos[1]); // dr = r1 - r0

// place back particle at COM
{
array1<double>::opt pos = motherPos[0];
for (auto d : irange(0, DIM)) pos[d] += dr[d] * 0.5;
}

// Random displacement of both elements around center of mass (position of back particle)
binaryRandomReorientation(motherPos[1], motherPos[0], params.divisionR0);
}

/*!
 Compute the inter-cellular force for one cell-cell pair : LJ or soft core or HZ interactions between elements
 */
 void cellCellForce(const CellParams &paramsi,const CellParams &paramsj, const cellIDs &ci, const cellIDs &cj, const Array::array2<double> &ri,
    const Array::array2<double> &rj, Array::array2<double> &fi, Array::array2<double> &fj) {
double f0, phi;
double rij[MAX_DIM] = {0.0, 0.0, 0.0};

for (auto iElemId : irange(0, colony.paramCells[ci.type].numElems)) {
auto iElemType = colony.paramCells[ci.type].elemType[iElemId];
auto &pi = colony.paramElems[iElemType];
pi.updateCore(colony.sigmaCore[ci.type][ci.id][iElemId]);

for (auto jElemId : irange(0, colony.paramCells[cj.type].numElems)) {
auto jElemType = colony.paramCells[cj.type].elemType[jElemId];
auto &pj = colony.paramElems[jElemType];
pj.updateCore(colony.sigmaCore[cj.type][cj.id][jElemId]);

if (paramsForce->elemForce(pi, pj, container.box.distance(rij, ri[iElemId], rj[jElemId]), f0, phi,
                                paramsOut.energy_record, potentialEnergy)) { // rij = rj - ri
if (f0 < 0.0 || ci.type == cj.type){
for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fi[iElemId][d] += f0 * rij[d];
fj[jElemId][d] -= f0 * rij[d];
}
}else{
for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fi[iElemId][d] += paramsi.crossAdhesion * f0 * rij[d];
fj[jElemId][d] -= paramsj.crossAdhesion * f0 * rij[d];
}
}
}
}
}
}

/*!
Compute dissipative/random forces of cells
prefix same means within one cell
cellBack means the dissipation happened between cells and background
*/
void sameCellDisRanForce(const CellParams &params, const cellIDs &ci, const Array::array2<double> &ri,
                Array::array2<double> &fid, Array::array2<double> &fir, Array::array2<double> &vi,
                Array::array1<double> &sigi) {
//double f0;
if (colony.paramCells[ci.type].numElems > 1) {
double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double nij[MAX_DIM] = {0.0, 0.0, 0.0};
double r_c = (colony.paramCells[ci.type].numElems == 1 ? colony.sigmaCore[ci.type][ci.id][0] : MIN(
colony.sigmaCore[ci.type][ci.id][0], colony.sigmaCore[ci.type][ci.id][1]));
double Gamma = paramsMD.dissamcel;
double Sigma = sqrt(2 * Gamma * paramsMD.kbt);

for (auto iiElemId : irange(0, colony.paramCells[ci.type].numElems)) {
auto iiElemType = colony.paramCells[ci.type].elemType[iiElemId];
auto &pi = colony.paramElems[iiElemType];
pi.updateCore(colony.sigmaCore[ci.type][ci.id][iiElemId]);

for (auto jjElemId : irange(iiElemId + 1, colony.paramCells[ci.type].numElems)) {
auto jjElemType = colony.paramCells[ci.type].elemType[jjElemId];
auto &pj = colony.paramElems[jjElemType];
pj.updateCore(colony.sigmaCore[ci.type][ci.id][jjElemId]);

auto r = sqrt(container.box.distance(rij, ri[iiElemId], ri[jjElemId]));
for (auto d : irange(0, DIM)) nij[d] = rij[d] / r;
auto wij = 1 - (r / r_c);

auto vii = vi[iiElemId];
auto vij = vi[jjElemId];

for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fid[iiElemId][d] = -Gamma * wij * ((vii[d] - vij[d]) * nij[d]) * nij[d];
fid[jjElemId][d] = -Gamma * wij * ((vij[d] - vii[d]) * (-nij[d])) * (-nij[d]);
auto rnd = Random::normal(0, 1);
fir[iiElemId][d] = Sigma * (sqrt(wij)) * nij[d] * rnd;
fir[jjElemId][d] = Sigma * (sqrt(wij)) * (-nij[d]) * rnd;
}
}
}
} else {
//for single-element cell, set the fid and fir as zero
for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fid[0][d] = 0;
fir[0][d] = 0;
}
}
}

void cellBackDisForce(const CellParams &params, const cellIDs &ci, const Array::array2<double> &ri,
           Array::array2<double> &fi, Array::array2<double> &vi) {
double Gamma = paramsMD.disbg;
for (auto iiElemId : irange(0, colony.paramCells[ci.type].numElems)) {
auto iiElemType = colony.paramCells[ci.type].elemType[iiElemId];
auto &pi = colony.paramElems[iiElemType];
pi.updateCore(colony.sigmaCore[ci.type][ci.id][iiElemId]);

for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fi[iiElemId][d] -= Gamma * vi[iiElemId][d];
}
}
}

void cellCellDisRanForce(const CellParams &params, const cellIDs &ci, const cellIDs &cj, const Array::array2<double> &ri,
       const Array::array2<double> &rj, Array::array2<double> &fid, Array::array2<double> &fjd,
       Array::array2<double> &fir, Array::array2<double> &fjr, Array::array2<double> &vi,
       Array::array2<double> &vj) {
//double f0

double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double nij[MAX_DIM] = {0.0, 0.0, 0.0};
double r_c = (colony.paramCells[ci.type].numElems == 1 ? colony.sigmaCore[ci.type][ci.id][0] : MIN(
colony.sigmaCore[ci.type][ci.id][0], colony.sigmaCore[ci.type][ci.id][1]));
double Gamma = paramsMD.disdifcel;
double Sigma = sqrt(2 * Gamma * paramsMD.kbt);

// calculate the dissipative force of elements from different cells
for (auto iElemId : irange(0, colony.paramCells[ci.type].numElems)) {
auto iElemType = colony.paramCells[ci.type].elemType[iElemId];
auto &pi = colony.paramElems[iElemType];
pi.updateCore(colony.sigmaCore[ci.type][ci.id][iElemId]);

for (auto jElemId : irange(0, colony.paramCells[cj.type].numElems)) {
auto jElemType = colony.paramCells[cj.type].elemType[jElemId];
auto &pj = colony.paramElems[jElemType];
pj.updateCore(colony.sigmaCore[cj.type][cj.id][jElemId]);

auto r = sqrt(container.box.distance(rij, ri[iElemId], rj[jElemId]));
for (auto d : irange(0, DIM)) nij[d] = rij[d] / r;
if (r < r_c) {
auto wij = 1 - (r / r_c);

auto vii = vi[iElemId];
auto vjj = vj[jElemId];

for (auto d : irange(0, DIM)) {
auto rnd = Random::normal(0, 1);
//#pragma omp atomic
fid[iElemId][d] -= Gamma * wij * ((vii[d] - vjj[d]) * nij[d]) * nij[d];
fjd[jElemId][d] -= Gamma * wij * ((vjj[d] - vii[d]) * (-nij[d])) * (-nij[d]);

fir[iElemId][d] += Sigma * sqrt(wij) * nij[d] * rnd;
fjr[jElemId][d] += Sigma * sqrt(wij) * (-nij[d]) * rnd;
}
} else return;
}
}
}

void sameCellDisForce(const CellParams &params, const cellIDs &ci, const Array::array2<double> &ri,
           Array::array2<double> &fi, Array::array2<double> &vi) {
//double f0;
if (colony.paramCells[ci.type].numElems > 1) {
double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double nij[MAX_DIM] = {0.0, 0.0, 0.0};
double r_c = (colony.paramCells[ci.type].numElems == 1 ? colony.sigmaCore[ci.type][ci.id][0] : MIN(
colony.sigmaCore[ci.type][ci.id][0], colony.sigmaCore[ci.type][ci.id][1]));
double Gamma = paramsMD.dissamcel;

for (auto iiElemId : irange(0, colony.paramCells[ci.type].numElems)) {
auto iiElemType = colony.paramCells[ci.type].elemType[iiElemId];
auto &pi = colony.paramElems[iiElemType];
pi.updateCore(colony.sigmaCore[ci.type][ci.id][iiElemId]);

for (auto jjElemId : irange(iiElemId + 1, colony.paramCells[ci.type].numElems)) {
auto jjElemType = colony.paramCells[ci.type].elemType[jjElemId];
auto &pj = colony.paramElems[jjElemType];
pj.updateCore(colony.sigmaCore[ci.type][ci.id][jjElemId]);

auto r = sqrt(container.box.distance(rij, ri[iiElemId], ri[jjElemId]));
for (auto d : irange(0, DIM)) nij[d] = rij[d] / r;
auto wij = 1 - (r / r_c);

auto vii = vi[iiElemId];
auto vij = vi[jjElemId];

for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fi[iiElemId][d] = -Gamma * wij * ((vii[d] - vij[d]) * nij[d]) * nij[d];
fi[jjElemId][d] = -Gamma * wij * ((vij[d] - vii[d]) * (-nij[d])) * (-nij[d]);
}
}
}
} else {
//for single-element cell, set the fid and fir as zero
for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fi[0][d] = 0;
fi[0][d] = 0;
}
}
}

void cellCellDisForce(const CellParams &params, const cellIDs &ci, const cellIDs &cj, const Array::array2<double> &ri,
           const Array::array2<double> &rj, Array::array2<double> &fi, Array::array2<double> &fj,
           Array::array2<double> &vi, Array::array2<double> &vj) {
//double f0;
double rij[MAX_DIM] = {0.0, 0.0, 0.0};
double nij[MAX_DIM] = {0.0, 0.0, 0.0};
double r_c = (colony.paramCells[ci.type].numElems == 1 ? params.sigmaCore[0] : MIN(params.sigmaCore[0],
                                                                                                                                        params.sigmaCore[1]));
double Gamma = paramsMD.disdifcel;
// calculate the dissipative force of elements from different cells
for (auto iElemId : irange(0, colony.paramCells[ci.type].numElems)) {
auto iElemType = colony.paramCells[ci.type].elemType[iElemId];
auto &pi = colony.paramElems[iElemType];
pi.updateCore(colony.sigmaCore[ci.type][ci.id][iElemId]);

for (auto jElemId : irange(0, colony.paramCells[cj.type].numElems)) {
auto jElemType = colony.paramCells[cj.type].elemType[jElemId];
auto &pj = colony.paramElems[jElemType];
pj.updateCore(colony.sigmaCore[cj.type][cj.id][jElemId]);

auto r = sqrt(container.box.distance(rij, ri[iElemId], rj[jElemId]));
for (auto d : irange(0, DIM)) nij[d] = -rij[d] / r;
if (r < r_c) {
auto wij = 1 - (SQ(r) / r_c);
auto vii = vi[iElemId];
auto vjj = vj[jElemId];
for (auto d : irange(0, DIM)) {
//#pragma omp atomic
fi[iElemId][d] -= Gamma * wij * ((vii[d] - vjj[d]) * nij[d]) * nij[d];
fj[jElemId][d] -= Gamma * wij * ((vjj[d] - vii[d]) * (-nij[d])) * (-nij[d]);
}
} else return;
}
}
}