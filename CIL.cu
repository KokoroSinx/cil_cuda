#include "CIL.hpp"
#include <vector>
#include <list>
#include <fstream>
#include <tuple>

using namespace std;

/*! \file CIL.cpp
 \brief Main loop of the simulation and functions working on the whole system, such as calculation of all the forces and implementation of integrators
 */

enum_set<OPT::OPT> SWITCH;           // Common.hpp
int DIM;              // Common.hpp
int ACTIVE_NOISE_DIM; // Common.hpp
Units units;       // Units.hpp
CellColony colony;      // Input.hpp
Container container;   // MD.hpp
MDParams paramsMD;    // MD.hpp
WallParams paramsWall;  // Wa  ll.hpp
OutputParams paramsOut;   // Input.hpp
PairForceParams *paramsForce; // MD.hpp
MDIntegrator mdStep;



/*!
 \brief Compute Intra-cellular CONSERVATIVE forces : Motility + FENE + Random forces
 */
void calculIntraForces(const vector <array3<double>> &positions, const vector <array3<double>> &forces,
											 const vector <array3<double>> &velocities, vector <array1<double>> &pressures) {
	for (auto ispec : irange(0, colony.getCellTypes())) { // over species
		const auto &pos = positions[ispec];
		const auto &frc = forces[ispec];
		const auto &vel = velocities[ispec];
		const auto &sig = colony.sigmaCore[ispec];
		const auto &pre = pressures[ispec];
		const auto &params = colony.paramCells[ispec];
		const auto nCells = colony.getNumCells(ispec);

		// Compute migration forces
		{
			auto &migrationFunc = colony.migrationFunc[ispec];
			if (migrationFunc != NULL) {
				const auto &state = colony.cellState[ispec];
				const auto &wiener = colony.activeWiener[ispec];
#pragma omp parallel for
				for (auto i = 0; i < nCells; i++)
					migrationFunc(params, state[i], pos[i], frc[i], vel[i], wiener[i], sig[i]);
			} else {
#pragma omp parallel for
				for (auto i = 0u; i < frc.Size(); i++) frc(i) = 0.0;
			}
		}
		// Compute External driving forces
		if (params.EXTERNAL) {
			const auto nElems = colony.getNumElems(ispec);
			auto frc_elems = array2<double>(nElems, DIM, frc());
#pragma omp parallel for
			for (auto i = 0; i < nElems; i++) {
				array1<double>::opt fi = frc_elems[i];
				for (auto d : irange(0, DIM)) {
#pragma omp atomic
					//fi[d] += params.fExt[d];
					fi[d] += 0;
				}
			}
		}
		// Compute Wall forces
		if (SWITCH[OPT::WALLS_ON]) {
#pragma omp parallel for
			for (auto i = 0; i < nCells; i++) {
				const array2<double> &cellFrc = frc[i];
				const array2<double> &cellPos = pos[i];
				double &cellPre = pre[i];
				const array1<double>::opt cellDia = sig[i];
				for (auto j : irange(0, params.numElems))
					for (auto &wall : paramsWall.walls)
						wall->addForce(cellDia[j], cellPos[j], cellFrc[j], cellPre);
			}
		}

		// Compute Passive Random forces
		if (SWITCH[OPT::NOISE_ON]) {
			const auto &wiener = colony.passiveWiener[ispec];
#pragma omp parallel for
			for (auto i = 0; i < nCells; i++) {
				const array2<double> &cellFrc = frc[i];
				array1<double>::opt rndFrc = wiener[i];
				for (auto j : irange(0, params.numElems)) {
					array1<double>::opt fij = cellFrc[j];
					for (auto d : irange(0, DIM)) {
//#pragma omp atomic
						fij[d] += params.zeta[j] * rndFrc[d];
					}
				}
			}
		}
	}// Over species
}



/*!
\brief Compute Inter-cellular forces : LJ or soft core interactions + OPTIONAL(Dis/Ran between elements)
*/

void calculInterForces(const vector <array3<double>> &positions, const vector <array3<double>> &forces,
											 const vector <array3<double>> &forces2, const vector <array3<double>> &forces3,
											 const vector <array3<double>> &velocities, vector <array1<double>> &pressures) {
	if (SWITCH[OPT::FORCE_GHOSTS]) return;
	// reset the pressures as zero before calculating pressure
	int nCells = colony.getTotalCells();
#pragma omp parallel for
	for (auto i = 0; i < nCells; i++) {
		auto ci = colony.UID2Cell(i);
		pressures[ci.type][ci.id] = 0;
	}
	if (SWITCH[OPT::LINKLIST_STENCIL]) {
		auto &lnk = container.linkList;
		auto maxCells = lnk.populate(colony);
		// int nCells = colony.getTotalCells();
		// add cells to link list, note that cells can denote both simulated cells and link list cells (maxCells means link list cells here, which are also called boxes)
#pragma omp parallel for
		for (auto iic = 0; iic < maxCells; iic++) {// over boxes
			int ic = (SWITCH[OPT::LINKLIST_SORTED] ? lnk.ihead[iic] : iic);
			int icell[DIM];
			lnk.link->get_cellCoord(icell, ic);
			for (auto dc = 0; dc < lnk.numNeighbors; dc++) { // over neighbor boxes
				auto jc = lnk.link->get_cellID(icell, lnk.neighborIds[dc]); //neighbor cell id
				if (jc == -1) continue; // Invalid Neighbor Box for non-pbc boundaries

				auto ii = lnk.head[ic];
				while (ii >= 0) { // over i cells in center box
					auto ci = colony.UID2Cell(ii);
					auto ri = positions[ci.type][ci.id];
					auto fi = forces[ci.type][ci.id];
					auto fid = forces2[ci.type][ci.id];
					auto fir = forces3[ci.type][ci.id];
					auto vi = velocities[ci.type][ci.id];
					auto &pri = pressures[ci.type][ci.id];
					const auto &params = colony.paramCells[ci.type];
					const auto &paramsi = colony.paramCells[ci.type];

					auto jj = lnk.head[jc];
					while (jj >= 0) { // over j cells in neighbor boxes
						auto cj = colony.UID2Cell(jj);
						jj = lnk.list[jj];
						if ((ci.type == cj.type && ci.type == 1) && (dc >= 14)) continue;
						if ((ci.type != cj.type) && (dc >= 448)) continue;
						auto rj = positions[cj.type][cj.id];
						auto fj = forces[cj.type][cj.id];
						auto fjd = forces2[cj.type][cj.id];
						auto fjr = forces3[cj.type][cj.id];
						auto vj = velocities[cj.type][cj.id];
						auto &prj = pressures[cj.type][cj.id];
						const auto &paramsj = colony.paramCells[cj.type];
						//
						if ((ic != jc || jj > ii) && ci != cj) {
							//cellCellForce_shortest(ci, cj, ri, rj, fi, fj);
							//cellCellForce_4r1a(params, ci, cj, ri, rj, fi, fj,pri, prj);
							//both spherecial shapes:
							//if ((paramsi.shape==paramsj.shape)&&(paramsi.shape==0)) cellCellForce(params, ci, cj, ri, rj, fi, fj, pri, prj);
							//cellCellForcePressureTestConstHz(params, ci, cj, ri, rj,fi,fj,pri, prj);
							cellCellForcePressure(paramsi, paramsj, ci, cj, ri, rj, fi, fj, pri, prj);
							//both rod shapes:
							//if ((paramsi.shape==paramaj.shape)&&(paramsi.shape==1)) rodRodForce(params, ci, cj, ri, rj, fi, fj, pri, prj);
							if (SWITCH[OPT::INTEGRATOR_DPD]) cellCellDisRanForce(params, ci, cj, ri, rj, fid, fjd, fir, fjr, vi, vj);
						}// interacting pairs
						jj = lnk.list[jj];
					}//j
					ii = lnk.list[ii];
				}//i
			}//dc
		}//ic
	}
	else if (SWITCH[OPT::LINKLIST_ON]) {
		auto &lnk = container.linkList;
		// TODO Check if the cell sizes are still small enough so that the linklist correctly catches all interactions.
		auto maxCells = lnk.populate(colony);
		// int nCells = colony.getTotalCells();
		// add cells to link list, note that cells can denote both simulated cells and link list cells (maxCells means link list cells here, which are also called boxes)
#pragma omp parallel for
		for (auto iic = 0; iic < maxCells; iic++) {// over boxes
			int ic = (SWITCH[OPT::LINKLIST_SORTED] ? lnk.ihead[iic] : iic);
			int icell[DIM];
			lnk.link->get_cellCoord(icell, ic);
			for (auto dc : irange(0, lnk.numNeighbors)) {  // over neighbor boxes
				auto jc = lnk.link->get_cellID(icell, lnk.neighborIds[dc]); //neighbor cell id
				if (jc == -1) continue; // Invalid Neighbor Box for non-pbc boundaries
				auto ii = lnk.head[ic];
				while (ii >= 0) { // over i cells in center box
					auto ci = colony.UID2Cell(ii);
					auto ri = positions[ci.type][ci.id];
					auto fi = forces[ci.type][ci.id];
					auto fid = forces2[ci.type][ci.id];
					auto fir = forces3[ci.type][ci.id];
					auto vi = velocities[ci.type][ci.id];
					auto &pri = pressures[ci.type][ci.id];
					const auto &params = colony.paramCells[ci.type];
					const auto &paramsi = colony.paramCells[ci.type];

					auto jj = lnk.head[jc];
					while (jj >= 0) { // over j cells in neighbor boxes
						auto cj = colony.UID2Cell(jj);
						auto rj = positions[cj.type][cj.id];
						auto fj = forces[cj.type][cj.id];
						auto fjd = forces2[cj.type][cj.id];
						auto fjr = forces3[cj.type][cj.id];
						auto vj = velocities[cj.type][cj.id];
						auto &prj = pressures[cj.type][cj.id];
						const auto &paramsj = colony.paramCells[cj.type];

						if ((ic != jc || jj > ii) && ci != cj) {
							//cellCellForce_shortest(ci, cj, ri, rj, fi, fj);
							//cellCellForce_4r1a(params, ci, cj, ri, rj, fi, fj,pri, prj);
							//both spherecial shapes:
							//if ((paramsi.shape == paramsj.shape) && (paramsi.shape == 0))
							//cellCellForcePressureTestConstHz(params, ci, cj, ri, rj, fi, fj, pri, prj);
							cellCellForcePressure(paramsi, paramsj, ci, cj, ri, rj, fi, fj, pri, prj);
							//both rod shapes:
							//if ((paramsi.shape==paramaj.shape)&&(paramsi.shape==1)) rodRodForce(params, ci, cj, ri, rj, fi, fj, pri, prj);
							if (SWITCH[OPT::INTEGRATOR_DPD]) cellCellDisRanForce(params, ci, cj, ri, rj, fid, fjd, fir, fjr, vi, vj);
						}// interacting pairs
						jj = lnk.list[jj];
					}//j
					ii = lnk.list[ii];
				}//i
			}//dc
		}//ic
	}
	else {
		int nCells = colony.getTotalCells();
#pragma omp parallel for
		for (auto ii = 0; ii < nCells; ii++) {

			auto ci = colony.UID2Cell(ii);
			auto ri = positions[ci.type][ci.id];
			auto fi = forces[ci.type][ci.id];
			auto fid = forces2[ci.type][ci.id];
			auto fir = forces3[ci.type][ci.id];
			auto &pri = pressures[ci.type][ci.id];
			auto vi = velocities[ci.type][ci.id];
			auto sigi = colony.sigmaCore[ci.type][ci.id];
			const auto &paramsi = colony.paramCells[ci.type];

			for (auto jj : irange(ii + 1, nCells)) {
				auto cj = colony.UID2Cell(jj);
				auto rj = positions[cj.type][cj.id];
				auto fj = forces[cj.type][cj.id];
				auto fjd = forces2[cj.type][cj.id];
				auto fjr = forces3[cj.type][cj.id];
				auto &prj = pressures[cj.type][cj.id];
				auto vj = velocities[cj.type][cj.id];
				const auto &paramsj = colony.paramCells[cj.type];
				//cellCellForce_shortest(ci, cj, ri, rj, fi, fj, vi, vj);
				//cellCellForce_4r1a(params, ci, cj, ri, rj, fi, fj);
				//cellCellForcePressureTestConstHz(params, ci, cj, ri, rj, fi, fj, pri, prj);
				cellCellForcePressure(paramsi, paramsj, ci, cj, ri, rj, fi, fj, pri, prj);
				if (SWITCH[OPT::INTEGRATOR_DPD]) cellCellDisRanForce(paramsi, ci, cj, ri, rj, fid, fjd, fir, fjr, vi, vj);
			}
		}
	}
}



/*!
 \brief Compute Intra-cellular DPD forces : Dis/Ran (2P) + cellBackDis
 */
void calculDPDIntraForces(const vector <array3<double>> &positions, const vector <array3<double>> &forces,
													const vector <array3<double>> &forces2, const vector <array3<double>> &forces3,
													const vector <array3<double>> &velocities) {
	int nCells = colony.getTotalCells();
#pragma omp parallel for
	for (auto i = 0; i < nCells; i++) {
		auto ci = colony.UID2Cell(i);
		auto ri = positions[ci.type][ci.id];
		auto fid = forces2[ci.type][ci.id];
		auto fir = forces3[ci.type][ci.id];
		auto vel = velocities[ci.type][ci.id];
		auto sig = colony.sigmaCore[ci.type][ci.id];
		auto params = colony.paramCells[ci.type];
		sameCellDisRanForce(params, ci, ri, fid, fir, vel, sig);
		cellBackDisForce(params, ci, ri, fid, vel);
	}
}



/*!
 \brief Compute Dissipative forces including intra-cellular, inter-cellular and dissipation with the background.
 */
void calculDisForces(const vector <array3<double>> &positions, const vector <array3<double>> &forces2,
										 const vector <array3<double>> &velocities) {
	if (SWITCH[OPT::FORCE_GHOSTS]) return;
	if (SWITCH[OPT::LINKLIST_ON]) {
		auto &lnk = container.linkList;
		auto maxCells = lnk.populate(
				colony); // add cells to link list, note that cells can denote both simulated cells and link list cells (maxCells means link list cells here, which are also called boxes)

#pragma omp parallel for
		for (auto iic = 0; iic < maxCells; iic++) { // over boxes
			int ic = (SWITCH[OPT::LINKLIST_SORTED] ? lnk.ihead[iic] : iic);
			int icell[DIM];
			lnk.link->get_cellCoord(icell, ic);
			auto limit = (colony.getCellTypes() > 1 ? 666 : 14);
			for (auto dc : irange(0, limit)) { // over neighbor boxes
				auto jc = lnk.link->get_cellID(icell, lnk.neighborIds[dc]); //neighbor cell id
				if (jc == -1) continue; // Invalid Neighbor Box for non-pbc boundaries

				auto ii = lnk.head[ic];
				while (ii >= 0) { // over i cells in center box
					auto ci = colony.UID2Cell(ii);
					auto ri = positions[ci.type][ci.id];
					auto fi = forces2[ci.type][ci.id];
					auto vi = velocities[ci.type][ci.id];
					const auto &params = colony.paramCells[ci.type];
					sameCellDisForce(params, ci, ri, fi, vi);
					cellBackDisForce(params, ci, ri, fi, vi);

					auto jj = lnk.head[jc];
					while (jj >= 0) { // over j cells in neighbor boxes
						auto cj = colony.UID2Cell(jj);
						jj = lnk.list[jj];
						if ((ci.type == cj.type && ci.type == 1) && (dc >= 14)) continue;
						if ((ci.type != cj.type) && (dc >= 448)) continue;
						auto rj = positions[cj.type][cj.id];
						auto fj = forces2[cj.type][cj.id];
						auto vj = velocities[cj.type][cj.id];


						if ((ic != jc || jj > ii) && ci != cj) {
							cellCellDisForce(params, ci, cj, ri, rj, fi, fj, vi, vj);
						}// interacting pairs
					}//j
					ii = lnk.list[ii];
				}//i
			}//dc
		}//ic
	} else {
		int nCells = colony.getTotalCells();
#pragma omp parallel for
		for (auto ii = 0; ii < nCells; ii++) {
			auto ci = colony.UID2Cell(ii);
			auto ri = positions[ci.type][ci.id];
			auto fi = forces2[ci.type][ci.id];
			auto vi = velocities[ci.type][ci.id];
			const auto &params = colony.paramCells[ci.type];

			sameCellDisForce(params, ci, ri, fi, vi);
			cellBackDisForce(params, ci, ri, fi, vi);

			for (auto jj : irange(ii + 1, nCells)) {
				auto cj = colony.UID2Cell(jj);
				auto rj = positions[cj.type][cj.id];
				auto fj = forces2[cj.type][cj.id];
				auto vj = velocities[cj.type][cj.id];

				cellCellDisForce(params, ci, cj, ri, rj, fi, fj, vi, vj);
			}
		}
	}
}

/*!
 \brief Test whether cells are ready to start dividing, or finish dividing, and perform divisions as needed
 */
void calculDivisions(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		auto &func = colony.divisionFunc[ispec];
		auto &fail = colony.divisionFail[ispec];
		auto &check = colony.divisionCheck[ispec];
		if (func != NULL) { // dividing species?
			const auto nCells = colony.getNumCells(ispec);
			const auto &params = colony.paramCells[ispec];
			const auto &pos = colony.pos[ispec];
			const auto &state = colony.cellState[ispec];
			const auto &time = colony.cellTSwitch[ispec];
			const auto &t0 = colony.cellT0[ispec];
			const auto &sigmaCore = colony.sigmaCore[ispec];
			const auto &vel = colony.vel[ispec];
			const auto &fic = colony.frc[ispec];
			const auto &fid = colony.frcDis[ispec];
			const auto &fir = colony.frcRan[ispec];
			const auto &posCellsDiv = colony.posCellsDiv[ispec];
			const auto &nCellsDiv = colony.nCellsDiv[ispec];
#pragma omp parallel for
			for (auto i = 0; i < nCells; i++) {
				if (state[i] == CellState::Dividing) {
					if (check(params, pos[i], sigmaCore[i])) { // sucessful division
//#pragma omp atomic
						int inew = colony.addCell(ispec);
						func(0, 0, params.divisionR0, pos[i], pos[inew]);

						if (paramsOut.div_app_record) {
							colony.setNumCellsDiv(ispec, nCellsDiv + 1, ts);
							posCellsDiv[nCellsDiv - 1] = pos[inew];
						}

						t0[inew] = ts;
						t0[i] = ts;
						time[inew] = ts + randomTimeSpan(params.migrationTau);
						state[inew] = CellState::Crawling;
						sigmaCore[inew] = params.sigmaCore;

						for (int k = 0; k < 2; ++k) {
							container.box.resetVel(vel[inew][k]);
							container.box.resetVel(vel[i][k]);
							container.box.resetFrc(fic[inew][k], fid[inew][k], fir[inew][k]);
							container.box.resetFrc(fic[i][k], fid[i][k], fir[i][k]);
						}

						//If the cell is typeswitching, change the species of the daughter cell by incrementing the species type by 1 and copy the cell over to the new species vector.
						if (params.typeswitching) {
							auto newSpec = ispec + 1; //TODO Do some sophisticated checking and choosing
							int inew2 = colony.addCell(
									newSpec); // TODO Define an alternate addCell that accepts a position as well.

							// TODO Make this depend on type of cell. I.e. if multiple disks present, this needs to look different.
							for (auto d : irange(0, DIM)) colony.pos[newSpec][inew2](0, d) = pos[inew](0, d);

							colony.sigmaCore[newSpec][inew2] = colony.paramCells[newSpec].sigmaCore;
							colony.removeCell(ispec, inew);
						}
						time[i] += randomTimeSpan(params.migrationTau);
						state[i] = CellState::Crawling;
						sigmaCore[i] = params.sigmaCore;
					} else if (time[i] == ts) {// failed division
						fail(params, pos[i]); // reset separation to prevent acceleration from failed division

						for (int k = 0; k < 2; ++k) {
							container.box.resetVel(vel[i][k]);
							container.box.resetFrc(fic[i][k], fid[i][k], fir[i][k]);
						}

						time[i] += randomTimeSpan(params.migrationTau);
						state[i] = CellState::Crawling;
						sigmaCore[i] = params.sigmaCore;
					}
				} else {
					if (time[i] == ts) {
						state[i] = CellState::Dividing;
						time[i] += randomTimeSpan(params.divisionTau);
						for (int k = 0; k < 2; ++k) {
							container.box.resetVel(vel[i][k]);
							container.box.resetFrc(fic[i][k], fid[i][k], fir[i][k]);
						}
					}
				}
			}
		}
	}
	colony.synchronize();
}



/*!
 \brief Defines the increment of the cell cycle phase at each time step. Since it only concerns itself with the appropriate reshaping of the cell cycle activity r, it has to be multiplied with the angular velocity variable to yield the correct final result. Based on the cell cycle model paper, J. Li et al, PRX (2021).
 */
double cellCyclePhaseVelocity0(double r, double RMaxedInv) {
  return MIN(RMaxedInv * r, 1.0);
}

/*!
 \brief Test whether cells are ready to start dividing, or finish dividing, and perform divisions as needed. This is the special version for single particle cells. Based on the cell cycle model paper, J. Li et al, PRX (2021), this has also been extended to keep track of the integration of the ODEs of the cell-cycle dependent variables, theta, r, and v.
 */
void calculDivisionsSingle(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		auto &func = colony.divisionFunc[ispec];
		// auto &fail = colony.divisionFail[ispec]; // in the cell cycle model as of Jan '23, cell divisions never fail
		// auto &check = colony.divisionCheck[ispec]; // in the cell cycle model as of Jan '23, cell divisions never fail
		const auto nCells = colony.getNumCells(ispec);
		const auto &tPressure = colony.cellTPressure[ispec];
		const auto &pressure = colony.pressure[ispec];
		const auto &params = colony.paramCells[ispec];
		const auto &state = colony.cellState[ispec];
		const auto &theta = colony.cellTTheta[ispec];
		const auto &rho = colony.cellTRho[ispec];

		auto tau_r = params.cycleTauR; 
		auto tau_v = params.cycleTauV;
		auto tau_p = params.cycleTauP;
		double v_s = params.v_s;

		// UNDER DEVELOPMENT: To express Bb Washout, we modify the pressure term locally and ad-hoc in order to mimic reduce traction forces. This functionality has some overlap with the implementation of an osmotic pressure paramsMD.pi below. Need to make sure that we don't have duplicate functionality.
		double pressureShiftDynamic = params.cyclePShift * exp(-ts*params.cyclePShiftTauInv);

#pragma omp parallel for
		for (auto i = 0; i < nCells; i++) {
			// auto pri = pressure[i];
			if ((state[i] != CellState::Dividing) || (pressure[i] < params.apopPressure)) tPressure[i] = 0;
			else tPressure[i] += 1;
			if (func != NULL) { // dividing species?
				const auto &pos = colony.pos[ispec];
				const auto &time = colony.cellTSwitch[ispec];
				const auto &t0 = colony.cellT0[ispec];
				const auto &angVel = colony.cellAngVel[ispec];
				// const auto &angel = colony.cellAngel[ispec];
				const auto &sigmaCore = colony.sigmaCore[ispec];
				const auto &sigmaCoreT = colony.sigmaCoreT[ispec];
				auto &vel = colony.vel[ispec];
				const auto &fic = colony.frc[ispec];
				const auto &fid = colony.frcDis[ispec];
				const auto &fir = colony.frcRan[ispec];
				const auto &posCellsDiv = colony.posCellsDiv[ispec];
				const auto &nCellsDiv = colony.nCellsDiv[ispec];
				if (state[i] == CellState::Dividing) {
					// Integrate the ODEs for the cell cycle by one timestep
					// TODO: For now this implicitly uses Euler, but do this consistently with the integration scheme used for the mechanics, i.e. apply Heun if we use Heun for the other equations. A lot of this loop should be somehow abstracted away into functions, or at least made more readable. E.g. a cell_cycle_step that would take up the integration scheme and a right-hand-side function for the cell cycle ODEs.
					// Cell cycle phase
					auto pNonDim = MAX(0., pressure[i] + pressureShiftDynamic) / params.cycleP0;
					auto theta_prime = angVel[i] * cellCyclePhaseVelocity0(rho[i], params.cycleRMaxedInv);
					theta[i] += theta_prime;

					// Cell volume
					double vt = (DIM == 3 ? Sigma2Volume(sigmaCoreT[i][0]) : Sigma2Area(sigmaCoreT[i][0])); // TODO: Assign a volume or area function once at startup and do none of this funny business.
					double v = (DIM == 3 ? Sigma2Volume(sigmaCore[i][0]) : Sigma2Area(sigmaCore[i][0]));
					auto c0 = angVel[i] * log(2) + (params.v_1 - 1) / tau_v; // TODO: For this formula, the quiescent volume is assumed to be one and v_1 is calculated relative to that. Is this consistent with the other non-dimensionless calculations involving an explicit v_s?
					auto vt_prime = c0 * rho[i] - (((vt - v_s) / v_s) / tau_v); 
					vt += vt_prime * vt;
					auto v_prime = -(v - (vt / (1 + params.betaVs * (paramsMD.pi + pNonDim)))) / tau_p;
					v += v_prime;
					v = tau_p == 0 ? vt : v;
					sigmaCoreT[i][0] = (DIM == 3 ? Volume2Sigma(vt) : Area2Sigma(vt));
					sigmaCore[i][0] = (DIM == 3 ? Volume2Sigma(v) : Area2Sigma(v));

					// Cell cycle activity
					auto r_prime_np = (1 - (rho[i]) - pNonDim);
					auto tau_r_pre = r_prime_np > 0 ? 1 : params.cycleRhoRatio;
					auto r_prime = r_prime_np / (tau_r * tau_r_pre);
					rho[i] += r_prime * rho[i];

					if (theta[i] >= 1) { // successful division
						int inew = colony.addCell(ispec);
						func(0, 0, sigmaCore[i][0], pos[i], pos[inew]);

						//todo write a function to make sure the new cell position is valid.
						assert(container.box.isValidPos(pos[i][0]));

						if (paramsOut.div_app_record) {
							colony.setNumCellsDiv(ispec, nCellsDiv + 1, ts);
							posCellsDiv[nCellsDiv - 1] = pos[inew];
						}
						double newSig;
						double newSigT;
						auto lMom = sigmaCore[i][0];
						auto lMomT = sigmaCoreT[i][0];
						if (DIM == 2) {
							newSig = lMom / sqrt(2);
							newSigT = lMomT / sqrt(2);
						} else if (DIM == 3) {
							auto rMom = (sigmaCore[i][0]) / 2;
							auto volumeMom = 4 * Constants::PI * rMom * rMom * rMom / 3;
							double a = 9 * lMom / 4;
							double b = -(3 * (lMom * lMom) / 2);
							double c = lMom * lMom * lMom / 4 - 3 * volumeMom / (Constants::PI);
							newSig = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
							auto rMomT = (sigmaCoreT[i][0]) / 2;
							auto volumeMomT = 4 * Constants::PI * rMomT * rMomT * rMomT / 3;
							a = 9 * lMomT / 4;
							b = -(3 * (lMomT * lMomT) / 2);
							c = lMomT * lMomT * lMomT / 4 - 3 * volumeMomT / (Constants::PI);
							newSigT = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
						}
						sigmaCore[inew][0] = newSig;
						sigmaCore[i][0] = newSig;
						sigmaCoreT[inew][0] = newSig; // TODO Should this be newSigT?
						sigmaCoreT[i][0] = newSig;		// TODO Should this be newSigT?

						t0[inew] = ts;
						t0[i] = ts;

						time[inew] = ts + randomTimeSpan(params.migrationTau);
						time[i] = ts + randomTimeSpan(params.migrationTau);

						state[i] = CellState::Dividing;
						state[inew] = CellState::Dividing;

						// angel[i] = 0;
						// angel[inew] = 0;

						theta[i] = 0;
						theta[inew] = 0;
						auto preTemp = pressure[i];
						pressure[inew] = preTemp;
						auto rhoTemp = rho[i];
						rho[inew] = rhoTemp;

						angVel[i] = 1.0 / randomTimeSpanLogNormal(params.divisionTau);//randomTimeSpan
						angVel[inew] = 1.0 / randomTimeSpanLogNormal(params.divisionTau);//randomTimeSpan

						for (int k = 0; k < params.numElems; ++k) {
							container.box.resetVel(vel[inew][k]);
							container.box.resetVel(vel[i][k]);
							container.box.resetFrc(fic[inew][k], fid[inew][k], fir[inew][k]);
							container.box.resetFrc(fic[i][k], fid[i][k], fir[i][k]);
						}

						//If the cell is typeswitching, change the species of the daughter cell by incrementing the species type by 1 and copy the cell over to the new species vector.
						if (params.typeswitching) {
							auto newSpec = ispec + 1; //TODO Do some sophisticated checking and choosing
							int inew2 = colony.addCell(
									newSpec); // TODO Define an alternate addCell that accepts a position as well.

							// TODO Make this depend on type of cell. I.e. if multiple disks present, this needs to look different.
							for (auto d : irange(0, DIM)) colony.pos[newSpec][inew2](0, d) = pos[inew](0, d);

							colony.sigmaCore[newSpec][inew2] = colony.paramCells[newSpec].sigmaCore;
							colony.removeCell(ispec, inew);
						}
					}
				}
			}
		}
	}
    colony.synchronize();
}



/*!
 \brief Test whether cells are dying
 Cells are randomly removed at a constant rate from the system
 */
void calculApoptosis(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &posCellsApp = colony.posCellsApp[ispec];
		const auto &nCellsApp = colony.nCellsApp[ispec];
		const auto &pos = colony.pos[ispec];
		const auto &params = colony.paramCells[ispec];
		if (params.APO) {
			int i = 0;
			auto nCells = colony.getNumCells(ispec);
			while (i < nCells) {
				if (Random::uniform0x(1.0) <= params.apopRate) {
					if (paramsOut.div_app_record) {
						colony.setNumCellsApp(ispec, nCellsApp + 1, ts);
						posCellsApp[nCellsApp - 1] = pos[i];
					}
					colony.removeCell(ispec, i);
					colony.synchronize();
					nCells = colony.getNumCells(ispec);
				} else i += 1;
			}
		}
	}
}

// Now we have a pressure-time-dependent apoptosis.
// Apoptosis cells are selected by after a DURATION(apopDuration) of HIGH PRESSURE(cellTPressure).
void calculApoptosisPressureTime(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &posCellsApp = colony.posCellsApp[ispec];
		const auto &nCellsApp = colony.nCellsApp[ispec];
		const auto &pos = colony.pos[ispec];
		const auto &params = colony.paramCells[ispec];
		const auto &tPressure = colony.cellTPressure[ispec];
		if (params.APO) {
			int i = 0;
			auto nCells = colony.getNumCells(ispec);
			while (i < nCells) {
				if (((params.apopDuration > 0) && (tPressure[i] >= (params.apopDuration / paramsMD.dt))) ||
						((params.apopRate > 0) && (Random::uniform0x(1.0) <= params.apopRate))) {
					if (paramsOut.div_app_record) {
						colony.setNumCellsApp(ispec, nCellsApp + 1, ts);
						posCellsApp[nCellsApp - 1] = pos[i];
					}
					colony.removeCell(ispec, i);
					colony.synchronize();
					nCells = colony.getNumCells(ispec);
				} else i += 1;
			}
		}
	}
}

// Now we have a cell cycle-dependent apoptosis.
// Apoptosis cells are selected by a function of r.
void calculApoptosisCycle(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &posCellsApp = colony.posCellsApp[ispec];
		const auto &nCellsApp = colony.nCellsApp[ispec];
		const auto &pos = colony.pos[ispec];
		const auto &params = colony.paramCells[ispec];
		const auto &rho = colony.cellTRho[ispec];
		if (params.APO) {
			int i = 0;
			auto nCells = colony.getNumCells(ispec);
			while (i < nCells) {
				if (rho[i] < 1) {
					auto app_rate = params.apopRate * pow((1 - (rho[i] / params.apopPressure)), params.apopDuration);
					if (Random::uniform0x(1.0) <= app_rate) {
						if (paramsOut.div_app_record) {
							colony.setNumCellsApp(ispec, nCellsApp + 1, ts);
							posCellsApp[nCellsApp - 1] = pos[i];
						}
						colony.removeCell(ispec, i);
						colony.synchronize();
						nCells = colony.getNumCells(ispec);
					} else i += 1;
				}
			}
		}
	}
}

// Now we have a cell cycle-dependent apoptosis.
// Apoptosis cells are selected by a function of r.
void calculApoptosisCyclePressure(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &posCellsApp = colony.posCellsApp[ispec];
		const auto &nCellsApp = colony.nCellsApp[ispec];
		const auto &pos = colony.pos[ispec];
		const auto &params = colony.paramCells[ispec];
		const auto &rho = colony.cellTRho[ispec];
		const auto &pressure = colony.pressure[ispec];
		if (params.APO) {
			int i = 0;
			auto nCells = colony.getNumCells(ispec);

			while (i < nCells) {
				auto app_rate = params.apopRate * pow((1 - (rho[i] / params.apopPressure)), params.apopDuration) *
												pow(pressure[i] / (pressure[i] + params.cycleP0), params.extrGamma);
				if (Random::uniform0x(1.0) <= app_rate) {
					if (paramsOut.div_app_record) {
						colony.setNumCellsApp(ispec, nCellsApp + 1, ts);
						posCellsApp[nCellsApp - 1] = pos[i];
					}
					colony.removeCell(ispec, i);
					colony.synchronize();
					nCells = colony.getNumCells(ispec);
				} else i += 1;
			}
		}
	}
}

// Now we have a cell cycle-dependent apoptosis.
// We have an independent random parameter to specify the overall apoptosis, which may related to the DNA
// mutation and DNA damage repair sort of things, which is supposed to be a small value
// Beyond, cell extruding happens when the local pressure is high, which plays a crucial role during the
// cell competition, so I would propose the extruding rate is related to k_extrude*(p-p0) when r = 0.
// which means, only quiescent cells under high compression would results in extrusion.
// Hence, we clarify the two ways toward death for cells:
// 1. Naturally, random and non-selective apoptosis. Can maintain the homeostasis density.
// 2. Pressure-dependent extrusion happens only for quiescent cells.
void calculApoptosisCycleExtrude(const int &ts) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &posCellsApp = colony.posCellsApp[ispec];
		const auto &nCellsApp = colony.nCellsApp[ispec];
		const auto &pos = colony.pos[ispec];
		const auto &params = colony.paramCells[ispec];
		const auto &rho = colony.cellTRho[ispec];
		const auto &pressure = colony.pressure[ispec];
		if (params.APO) {
			const auto nCells = colony.getNumCells(ispec);
			for (auto i : irange(0, nCells)) {
				// for overall apoptosis
				if (rho[i] < 1) {
					auto app_rate = params.apopRate;
					if (Random::uniform0x(1.0) <= app_rate) {
						if (paramsOut.div_app_record) {
							colony.setNumCellsApp(ispec, nCellsApp + 1, ts);
							posCellsApp[nCellsApp - 1] = pos[i];
						}
						colony.removeCell(ispec, i);
						colony.synchronize();
					}
				}
				// for compaction-driven extruding
				if (rho[i] <= 0.1) {
					auto extr_rate =
							params.extrRate * pow(((pressure[i] - params.apopPressure) / params.apopPressure), params.extrGamma);
					if (Random::uniform0x(1.0) <= extr_rate) {
						if (paramsOut.div_app_record) {
							colony.setNumCellsApp(ispec, nCellsApp + 1, ts);
							posCellsApp[nCellsApp - 1] = pos[i];
						}
						colony.removeCell(ispec, i);
						colony.synchronize();
					}
				}
			}
		}
	}
}

/*!
 \brief Calculates the appropriate pseudo-random fluctuations for the Wiener process employed in the calculation of forces
 */
void updateWienerProcess() {
	if (SWITCH[OPT::NOISE_ON]) {
		// Passive Noise
		for (auto ispec: irange(0, colony.getCellTypes())) {
			const auto &params = colony.paramCells[ispec];
			const auto &wiener = colony.passiveWiener[ispec];
			if (params.dNoise[NOISE::PASSIVE] > 0.0)
				for (auto i : irange(0u, wiener.Size())) wiener(i) = params.dNoise[NOISE::PASSIVE] * Random::normal();
		}
	}
	if (SWITCH[OPT::NOISE_ACTIVE]) {
		// Active Noise
		for (auto ispec: irange(0, colony.getCellTypes())) {
			const auto &params = colony.paramCells[ispec];
			if (params.numElems > 1) {
				const auto &wiener = colony.activeWiener[ispec];
				const auto nCells = colony.getNumCells(ispec);

				if (params.dNoise[NOISE::RADIAL] > 0.0)
					for (auto i : irange(0, nCells)) wiener(i, 0) = params.dNoise[NOISE::RADIAL] * Random::normal();
				if (params.dNoise[NOISE::ANGULAR] > 0.0)
					for (auto i : irange(0, nCells)) wiener(i, 1) = params.dNoise[NOISE::ANGULAR] * Random::normal();
			}
		}
	}
}


void calculPressure(const vector <array3<double>> &forces, const vector <array3<double>> &forces2,
							 vector <array1<double>> &pressures) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &params = colony.paramCells[ispec];
		const auto nCells = colony.getNumCells(ispec);
		auto &frc = forces[ispec];
		auto &frc2 = forces2[ispec];
		auto &pre = pressures[ispec];

#pragma omp parallel for
		for (auto i = 0; i < nCells; i++) {
			const auto &Fi = frc[i];
			const auto &Fid = frc2[i];
			auto &Pi = pre[i];
			for (auto j : irange(0, params.numElems)) {
				auto zeta = params.zeta[j];
				auto sigmai = colony.sigmaCore[ispec][i][j];
				auto areai = sigmai * Constants::PI;
				auto Fi_total = 0;
				for (auto d: irange(0, DIM)) {
					Fi_total += Fi[j][d] + Fid[j][d];
				}
				Pi = Fi_total / areai;
			}
		}
	}
}

void updatePositions(const vector <array3<double>> &positions, const vector <array3<double>> &forces, const double &deltaT) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &params = colony.paramCells[ispec];
		const auto nCells = colony.getNumCells(ispec);
		auto &frc = forces[ispec];
		auto &pos = positions[ispec];
#pragma omp parallel for
		for (auto i = 0; i < nCells; i++) {
			const auto &Ri = pos[i];
			const auto &Fi = frc[i];
			for (auto j : irange(0, params.numElems)) {
				auto temp = (SWITCH[OPT::INTEGRATOR_DPD] ? deltaT : deltaT * params.zetaInv[j]);
				if (!SWITCH[OPT::INTEGRATOR_DPD]) run_assert(params.zeta[j], "In Euler, zeta should > 0!!!");
				container.box.updatePos(Ri[j], Fi[j], temp);
			}
		}
	}
}



// Update velocities for DPD method.
void updateVelocities(const vector <array3<double>> &velocities, const vector <array3<double>> &forces,
											const vector <array3<double>> &forces2, const vector <array3<double>> &forces3,
											const double &deltaT) {
#pragma omp parallel for
	for (auto ispec = 0; ispec < colony.getCellTypes(); ispec++) {
		const auto &params = colony.paramCells[ispec];
		const auto nCells = colony.getNumCells(ispec);
		auto &vel = velocities[ispec];
		auto &frc = forces[ispec];
		auto &frcDis = forces2[ispec];
		auto &frcRan = forces3[ispec];
		auto &dt = deltaT;

#pragma omp parallel for
		for (auto i = 0; i < nCells; i++) {
			const auto &vi = vel[i];
			const auto &fic = frc[i];
			const auto &fid = frcDis[i];
			const auto &fir = frcRan[i];

			for (auto j : irange(0, params.numElems)) {
				auto &ms = params.ms[j];
				//Note: in the integration of DPD, velocity is updated twice using half time step.
				container.box.updateVel(vi[j], fic[j], fid[j], fir[j], dt, 2 * ms);
			}
		}
	}
}


// Update the velocities and positions. For Euler method.
void updateVelocityPositions(const int ts, const vector <array3<double>> &positions, const vector <array3<double>> &forces,
												const vector <array3<double>> &velocities, vector <array1<double>> &pressures,const double &deltaT) {
	for (auto ispec: irange(0, colony.getCellTypes())) {
		const auto &params = colony.paramCells[ispec];
		auto p0 = params.cycleP0;
		const auto nCells = colony.getNumCells(ispec);
		const auto &t0 = colony.cellT0[ispec];
		auto &frc = forces[ispec];
		auto &pos = positions[ispec];
		auto &vel = velocities[ispec];
		const auto &pre = pressures[ispec];
#pragma omp parallel for
		for (auto i = 0; i < nCells; i++) {
			const auto &Ri = pos[i];
			const auto &Fi = frc[i];
			const auto &vi = vel[i];
			const auto &pi = pre[i];
			const auto &sigmaCore = colony.sigmaCore[ispec];
			for (auto j : irange(0, params.numElems)) {
				auto area_frc = Sigma2Area(sigmaCore[i][0]);
				auto pNonDim = pi/p0;
				auto temp = (SWITCH[OPT::INTEGRATOR_DPD] ? deltaT : deltaT * 1/((params.zeta(j)+params.zeta_p(j)*pow(pNonDim,params.zeta_p_beta(j)))*area_frc));
				if (!SWITCH[OPT::INTEGRATOR_DPD]) run_assert(params.zeta(j), "In Euler, zeta should > 0!!!");
                long double absVijTemp = 0;
                long double absFijTemp = 0;
                for (auto d : irange(0, DIM)) {
                    absVijTemp += vi[j][d]*vi[j][d];
                    absFijTemp += Fi[j][d]*Fi[j][d];
                }
                auto absVij = sqrt(absVijTemp);
                auto absFij = sqrt(absFijTemp);
				temp = 0  ? (absVij <= params.static_vel(j) && absFij <= params.static_force(j)) : temp;
				container.box.updatePos(Ri[j], Fi[j], temp);
				container.box.updateVel(vi[j], Fi[j], temp / deltaT);
			}
		}
	}
}


/*!
 \brief Tests whether all particles are still inside the container
 */
void checkPositions(const vector <array3<double>> &positions) {
#pragma omp parallel for
	for (auto ispec = 0; ispec < colony.getCellTypes(); ispec++) {
		const auto &params = colony.paramCells[ispec];
		auto nCells = colony.getNumCells(ispec);
		auto &pos = positions[ispec];
//#pragma omp parallel for
		int i = 0;
		while (i < nCells) {
			const auto &Ri = pos[i];
			for (auto j : irange(0, params.numElems)) {
				if (!paramsMD.drain) run_assert(container.box.isBoundedPos(Ri[j]), "Particle outside domain!");
				else if (!container.box.isBoundedPos(Ri[j])) {
					colony.removeCell(ispec, i);
					colony.synchronize();
					nCells = colony.getNumCells(ispec);
					break;
				} else continue;
			}i += 1;
		}
	}
}
/*!
\brief Modify the element diameters to account for cell growth during cell divisions
 */
inline void updateDiameters(int ts, CellColony &colony) {
#pragma omp parallel for
	for (auto ispec = 0; ispec < colony.getCellTypes(); ispec++) {
		const auto &params = colony.paramCells[ispec];
		if (params.DIV) {
			const auto &state = colony.cellState[ispec];
			const auto &sigmaCore = colony.sigmaCore[ispec];
			const auto nCells = colony.getNumCells(ispec);
			const double maxSigmaCore = params.sigmaCore.Max();
#pragma omp parallel for
			for (auto i = 0; i < nCells; i++) {
				if (state[i] == CellState::Dividing && params.divisionSwelling) {
					for (auto j : irange(0, params.numElems)) {
						// Scale the diameter linearly over time
						if (sigmaCore[i][j] < maxSigmaCore)
							sigmaCore[i][j] += (maxSigmaCore - sigmaCore[i][j]) / (colony.cellTSwitch[ispec][i] - ts);
					} // j
				} 
			}// i
		} 
	}//ispec
}


/*!
\brief Update forces for all methods. Calculation is based on the iteration method.
 All forces are left space here, while only in DPD, force2&force3 are using.
 TODO: Consider a more memory-efficient method to handle the force calculation.
 */
inline void calculForcesPressure(const vector <array3<double>> &positions, const vector <array3<double>> &forces,
																 const vector <array3<double>> &forces2, const vector <array3<double>> &forces3,
																 const vector <array3<double>> &velocities, vector <array1<double>> &pressures,
																 const bool &drawRandom = true) {
	if (drawRandom) updateWienerProcess();
	calculIntraForces(positions, forces, velocities, pressures);
	//if (SWITCH[OPT::INTEGRATOR_DPD]) calculDPDIntraForces(positions, forces, forces2, forces3, velocities);
	calculInterForces(positions, forces, forces2, forces3, velocities, pressures);
}


/*!
 \brief Integrator for the Euler method, propagates the equations of motion by one timestep
 */
void Euler(int &ts) {
	updateVelocityPositions(ts, colony.pos, colony.frc, colony.vel,colony.pressure, paramsMD.dt);
    colony.calculComs();
    calculDivisionsSingle(++ts);
    calculApoptosisCyclePressure(ts);
    updateDiameters(ts, colony);
    calculForcesPressure(colony.pos, colony.frc, colony.frcDis, colony.frcRan, colony.vel, colony.pressure);
}




/*!
 \brief Integrator for the Stochastic Heun method, propagates the equations of motion by one timestep
 */
void StochasticHeun(int &ts) {
	for (auto i : irange(0, colony.getCellTypes())) {
		auto &x = colony.pos[i];
		auto &xcpy = colony.pos0[i];
//#pragma omp parallel for
		for (auto j : irange(0u, x.Size())) xcpy(j) = x(j);
	}
	//predictor
	updatePositions(colony.pos0, colony.frc, paramsMD.dt);
	updatePositions(colony.pos, colony.frc, paramsMD.hdt);

	//corrector
	calculForcesPressure(colony.pos, colony.frc, colony.frcDis, colony.frcRan, colony.vel, colony.pressure);
	updatePositions(colony.pos, colony.frc, paramsMD.hdt);

	colony.calculComs();
	calculDivisions(++ts);
	calculApoptosis(ts);
	updateDiameters(ts, colony);

	calculForcesPressure(colony.pos, colony.frc, colony.frcDis, colony.frcRan, colony.vel, colony.pressure);
}

/*!
 \brief Integrator for the Dissipative Particle Dynamics method, propagates the equations of motion by one timestep
 */
void DPD(int &ts) {
	updateVelocities(colony.vel, colony.frc, colony.frcDis, colony.frcRan, paramsMD.dt);
	updatePositions(colony.pos, colony.vel, paramsMD.dt);

	calculForcesPressure(colony.pos, colony.frc, colony.frcDis, colony.frcRan, colony.vel, colony.pressure);
	updateVelocities(colony.vel, colony.frc, colony.frcDis, colony.frcRan, paramsMD.dt);

	calculDisForces(colony.pos, colony.frcDis, colony.vel);
	colony.calculComs();
	calculDivisionsSingle(++ts);
	calculApoptosisPressureTime(ts);
	//updateDiameters(ts, colony);
}

// Initializing potential energy here, it will be calculated at the place where potential forces calculated, in MD.cpp.
// Initializing cpuTime, record the time spend on each frame.
double potentialEnergy, cpuTime;

int main(int argc, char **argv) {
	init_threads();
	using std::cerr;
	using std::cout;
	using std::endl;

	if (argc < 2 or !fileCheck(argv[1])) {
		cerr << "Usage: ./CIL.x in.json" << endl;
		exit(1);
	}
	//cerr << "# Version no.: " << VERSION << std::endl;
	{ // Parse json input file & initialize
		json jiop;
		json_parser::parse_file(argv[1], jiop);
		initialize(jiop);
		json_parser::dump((paramsOut.dirName + "/RunParams/" + paramsOut.prjName + ".RunParams.json").c_str(), jiop);
	}
	{ // simulate
		cerr << fmtTitle % "Start Simulation" << endl;
		WallTimer timer, timer1;
		timer.start();
		int ts = 0;

		//Now I initialise all the forces for Euler and DPD, in the afterward iteration, corresponding forces(DPD or EULER)will be updated
		calculForcesPressure(colony.pos, colony.frc, colony.frcDis, colony.frcRan, colony.vel, colony.pressure);
        //calculForcesPressureTorquesRodDev(colony);
		outputFrame(0, 0);
		format line("# ts = %8d time = %12.3e \t cells = %8d elements = %8d\n");
		// Mechanism for detecting whether the program has received a kill signal: check for the existence of this file
		// const char *KILL = (paramsOut.dirName + "/RunParams/" + paramsOut.prjName + ".KILL").c_str();
		for (auto iframe : irange(1, paramsMD.frames + 1)) {
			//Initializing energy as ZERO in the beginning of each step, and reset the timer inside the loop.
			timer1.start();
			potentialEnergy = 0;
			for (auto its : irange(0, paramsMD.gts)) mdStep(ts);
			checkPositions(colony.pos);
            cerr << line % ts % (ts * paramsMD.dt) % colony.getTotalCells() % colony.getTotalElems();
			if (SWITCH[OPT::OUTPUT_DUMP]) outputFrame(iframe, ts);
//			if (fileCheck(KILL)) {
//				cerr << fmtTitle % "Received KILL signal" << endl;
//				break;
//			}
			cpuTime = timer1.stop();
			//iframe++;
		}
		{
			finalize();
		}
		cerr << fmtTitle % "End Simulation" << endl;
		double wtime = timer.stop();
		cerr << fmtDataDbl % "Time (seconds)" % wtime;
		cerr << fmtDataDbl0 % "(minutes)" % (wtime / 60.0);
		cerr << fmtDataDbl0 % "(days)" % (wtime / 3600.0 / 24.0);
		cerr << '\n';
	}
}