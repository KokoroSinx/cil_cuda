#include "Input.hpp"

/*!
 Read species parameters
 */
void readSpecs(const json &jiop) {
    std::cerr << "# Reading species data ...";
    colony.init(json_parser::get<vector < string>>
    (jiop, "DEFINE"));
    int elemID = 0;
    for (auto cellType : irange(0, colony.getCellTypes())) {
        auto &loc = json_parser::get_child(jiop, colony.nameCells[cellType]);
        CellParams &params = colony.paramCells[cellType];

        // Initialize
        params.init(elemID, json_parser::get<int>(loc, "NELEM"));

        auto names = json_parser::get_vector<string>(loc, "NAMES", params.numElems);
        for (auto i : irange(0, params.numElems)) params.mapNames.insert(std::pair<string, int>(names[i], i));
        run_assert(params.mapNames.size() == params.numElems, "element names must be unique!");
        params.shape = json_parser::get<bool>(loc, "SHAPE");
        params.CIL = (params.numElems > 1 ? json_parser::get<bool>(loc, "CIL") : false);
        params.crossAdhesion = json_parser::get<double>(loc, "CROSS_ADHESION");

        {
            json_parser::load_data(loc, "EXTFORCE", params.fExt(), DIM);
            params.fExt *= units.Force;
            auto temp = 0.0;
            for (auto d : irange(0, DIM)) temp += SQ(params.fExt[d]);
            if (non_zero_mp(temp)) params.EXTERNAL = true;
        }
        // Read element wise parameters
        {
            run_assert(params.numElems < 3, "Only binary cells implemented so far...\n");
            if (params.numElems == 2) {
                json_parser::load_data(loc, "GROWTHK0", params.growthk0(), params.numElems);
                json_parser::load_data(loc, "GROWTHK1", params.growthk1(), params.numElems);
                json_parser::load_data(loc, "GROWTHK2", params.growthk2(), params.numElems);
            }
            json_parser::load_data(loc, "MOTILITY", params.m(), params.numElems);
            json_parser::load_data(loc, "FRICTION", params.zeta(), params.numElems);
            json_parser::load_data(loc, "P_FRICTION", params.zeta_p(), params.numElems);
            json_parser::load_data(loc, "P_FRICTION_BETA", params.zeta_p_beta(), params.numElems);
            json_parser::load_data(loc, "FRICTION_STATIC_VEL", params.static_vel(), params.numElems);
            json_parser::load_data(loc, "FRICTION_STATIC_FORCE", params.static_force(), params.numElems);
            json_parser::load_data(loc, "MASS", params.ms(), params.numElems);
            // json_parser::load_data(loc, "ROD_RADIUS", params.rodRadius(), params.numElems);
            if (SWITCH[OPT::FORCE_SOFTCORE]) {
                auto sloc = json_parser::get_child(loc["PAIRFORCES"], "SC");
                json_parser::load_data(sloc, "SIGMA_MIN", params.sigmaMin(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_CORE", params.sigmaCore(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_THETA", params.sigmaTheta(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_MAX", params.sigmaMax(), params.numElems);
                json_parser::load_data(sloc, "EPSILON_CORE", params.epsilonCore(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_RATIO", params.sigmaWell(), params.numElems);
                json_parser::load_data(sloc, "EPSILON_RATIO", params.epsilonWell(), params.numElems);
                for (auto i : irange(0, params.numElems)) {
                    run_assert(params.sigmaWell[i] <= 1.0, "Sigma Ratio > 1 !!!");
                    params.sigmaWell[i] *= params.sigmaCore[i];
                }
                for (auto i : irange(0, params.numElems)) params.epsilonWell[i] *= params.epsilonCore[i];
                params.sigmaWell *= units.Length;
                params.epsilonCore *= units.Energy;
                params.epsilonWell *= units.Energy;
            } else if (SWITCH[OPT::FORCE_HERTZIAN]||SWITCH[OPT::FORCE_EA]) {
                auto sloc = json_parser::get_child(loc["PAIRFORCES"], "HZ");
                json_parser::load_data(sloc, "ELASTIC_MODULUS", params.epsilonCore(), params.numElems);
                json_parser::load_data(sloc, "POISSON_RATIO", params.sigmaWell(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_CORE", params.sigmaCore(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_THETA", params.sigmaTheta(), params.numElems);
                json_parser::load_data(sloc, "ADHESION_CONSTANT", params.epsilonWell(), params.numElems);
                params.epsilonCore *= units.Pressure;
                params.epsilonWell *= units.Force / SQ(units.Length);
            } else { // Default to Lennard-Jones
                auto sloc = json_parser::get_child(loc["PAIRFORCES"], "LJ");
                json_parser::load_data(sloc, "SIGMA", params.sigmaCore(), params.numElems);
                json_parser::load_data(sloc, "EPSILON", params.epsilonWell(), params.numElems);
                json_parser::load_data(sloc, "SIGMA_THETA", params.sigmaTheta(), params.numElems);
                params.sigmaWell *= units.Length;
                params.epsilonCore *= units.Energy;
                params.epsilonWell *= units.Energy;
            }
            params.m *= units.Motility;
            params.growthk0 *= units.Motility;
            params.growthk1 *= units.Motility;
            params.growthk2 *= units.Motility;
            params.zeta *= units.Friction;
						params.zeta_p *= units.Friction;
            params.ms *= units.Mass;
            params.sigmaMin *= units.Length;
            params.sigmaCore *= units.Length;
            params.sigmaTheta *= units.Angle;
            for (auto i : irange(0, params.numElems)) params.zetaInv[i] = 1.0 / params.zeta[i];
        }

        //Noise parameters for species
        {
            if (SWITCH[OPT::NOISE_ON]) {
                auto sloc = json_parser::get_child(loc, "NOISE");
                params.dNoise[NOISE::PASSIVE] = json_parser::get<double>(sloc, "PASSIVE");

                if (params.numElems > 1 && SWITCH[OPT::NOISE_ACTIVE]) { // active cell noise
                    params.dNoise[NOISE::RADIAL] = json_parser::get<double>(sloc, "RADIAL");
                    params.dNoise[NOISE::ANGULAR] = json_parser::get<double>(sloc, "ANGULAR");
                }
            }

            params.dNoise[NOISE::PASSIVE] *= units.SpatialDiffusion;
            params.dNoise[NOISE::RADIAL] *= units.SpatialDiffusion;
            params.dNoise[NOISE::ANGULAR] *= units.AngularDiffusion;

            params.dNoise[NOISE::PASSIVE] = sqrt(2.0 * params.dNoise[NOISE::PASSIVE]);
            params.dNoise[NOISE::RADIAL] = sqrt(2.0 * params.dNoise[NOISE::RADIAL]);
            params.dNoise[NOISE::ANGULAR] = sqrt(2.0 * params.dNoise[NOISE::ANGULAR]);
        }
        // Read bond parameters
        {
            if (params.numBonds > 0) { // read pair bonding parameters  : upper triangular matrix
                auto kij = json_parser::get_vector<double>(loc["FENE"], "KAPPA_PULL", params.numBonds);
                auto kij_push = json_parser::get_vector<double>(loc["FENE"], "KAPPA_PUSH", params.numBonds);
                auto spring_ratio = json_parser::get_vector<double>(loc["FENE"], "SPRING_RATIO", params.numBonds);
                auto rij = json_parser::get_vector<double>(loc["FENE"], "RMAX", params.numBonds);
                int shift = 0;
                int count = 0;
                for (auto i : irange(0, params.numElems - 1)) {
                    shift += i + 1; //index shift to upper diagonal element
                    for (auto j : irange(i + 1, params.numElems)) {
                        auto im = i * params.numElems + j - shift;
                        params.kappa(i, j) = params.kappa(j, i) = kij[im] * units.Motility;
                        params.kappa_push(i, j) = params.kappa_push(j, i) = kij_push[im] * units.Motility;
                        params.spring_ratio(i, j) = params.spring_ratio(j, i) = spring_ratio[im];
                        params.r2max(i, j) = params.r2max(j, i) = SQ(rij[im] * units.Length);
                        count++;
                    }
                }
                run_assert(count == params.numBonds, "Incorrect number of bonds!");
            }// FENE bonds

            // Compute steady-state distances
            if (params.numElems == 2) {
                double alpha = params.zetaInv[0] + params.zetaInv[1];
                double beta = params.m[0] * params.zetaInv[0] + params.m[1] * params.zetaInv[1];
                params.rss(0, 1) = params.rss(1, 0) = sqrt(
                        params.r2max(0, 1) * (1.0 - alpha / beta * params.kappa(0, 1)));
            }
        }

        // Read division parameters
        {
            //if(params.numElems > 1 && json_parser::get<bool>(loc["DIVISION"], "ENABLED")){
            if (json_parser::get<bool>(loc["DIVISION"], "ENABLED")) {

                auto dloc = json_parser::get_child(loc, "DIVISION");
                params.DIV = true;
							  params.divisionConstForce = json_parser::get<bool>(dloc, "CONST_FORCE");
                params.divisionSwelling = json_parser::get<bool>(dloc, "SWELLING");
                params.divisionR0 = json_parser::get<double>(dloc, "R0") * units.Length;
                params.divisionR2min = SQ(json_parser::get<double>(dloc, "Rmin") * units.Length);
                params.divisionR2minRatio = json_parser::get<double>(dloc, "Rmin_RATIO");
                params.divisionSigmaBirthRatio = json_parser::get<double>(dloc, "SIGMA_BIRTH_RATIO");
                params.divisionSigmaMin = json_parser::get<double>(dloc, "SIGMA_DIVISION_MIN") * units.Length;
                params.divisionTau.mu = json_parser::get<double>(dloc["TAU_DIVISION"], "AVG") * units.Time;
                // params.divisionPressure = json_parser::get<double>(dloc, "PRESSURE_THRESHOLD") * units.Pressure;
                params.divisionTau.sigma = json_parser::get<double>(dloc["TAU_DIVISION"], "SIGMA") * units.Time;
                params.migrationTau.mu = json_parser::get<double>(dloc["TAU_MIGRATION"], "AVG") * units.Time;
                params.migrationTau.sigma = json_parser::get<double>(dloc["TAU_MIGRATION"], "SIGMA") * units.Time;

                // params.cycleAlpha = json_parser::get<double>(dloc, "CYCLE_ALPHA");
                // params.cycleBeta = json_parser::get<double>(dloc, "CYCLE_BETA");
                params.v_1 = json_parser::get<double>(dloc, "V1");
                // TODO For now, some parts of the code implicitly assume that v_s = 1 always. Here, we implicitly assume this as well, since v_1 is defined in units of v_s. If we ever make v_s different from that, we need to rescale v_1 with something.
                // We should test that sigmaCore == cycleSigmaMin
                // == 1.1283791670955126
                // or make the code more general
                // params.cycleGamma = json_parser::get<double>(dloc, "CYCLE_GAMMA");
                params.cycleTauR = json_parser::get<double>(dloc, "CYCLE_TAU_R")* params.divisionTau.mu;
                params.cycleTauV = json_parser::get<double>(dloc, "CYCLE_TAU_V")* params.divisionTau.mu;
                params.cycleTauP = json_parser::get<double>(dloc, "CYCLE_TAU_P")* params.divisionTau.mu;

                params.cycleSigmaDiv = json_parser::get<double>(dloc, "CYCLE_SIGMA_DIV");
                double cycleSigmaMin = json_parser::get<double>(dloc, "CYCLE_SIGMA_MIN");
                params.v_s = (DIM == 3 ? Sigma2Volume(cycleSigmaMin) : Sigma2Area(cycleSigmaMin));

                params.cycleP0 = json_parser::get<double>(dloc, "CYCLE_P0");
                  if (json_parser::exists(dloc, "CYCLE_P_SHIFT")) {
                  params.cyclePShift =
                      json_parser::get<double>(dloc, "CYCLE_P_SHIFT")*params.cycleP0;
                }
                if (json_parser::exists(dloc, "CYCLE_P_SHIFT_TAU")) {
                  params.cyclePShiftTauInv =
                      1.0 /
                      (json_parser::get<double>(dloc, "CYCLE_P_SHIFT_TAU") *
                       params.divisionTau.mu);
                }
                params.betaVs = json_parser::get<double>(dloc, "CYCLE_BETA_VS");
                params.cycleRhoRatio = json_parser::get<double>(dloc, "CYCLE_RHO_RATIO");
                if (json_parser::exists(dloc, "CYCLE_R_MAXED")) {
                  params.cycleRMaxedInv =
                      1.0 / json_parser::get<double>(dloc, "CYCLE_R_MAXED");
                }

                if (json_parser::exists(dloc, "TYPESWITCHING")) {
                    params.typeswitching = json_parser::get<bool>(dloc, "TYPESWITCHING");
                    string switch_to = json_parser::get<string>(dloc, "SWITCH_TO");
                    int spec_i = colony.name2CellType(switch_to);
                    // auto &params_i = colony.paramCells[spec_i];
                    // TODO Save the cell type switch to parameters
                }

                // If a value for Rmax specific to the division state is set in the JSON file, read that. Otherwise, use the general value
                if (json_parser::exists(loc["FENE"], "RMAX_DIVISION")) {
                    if (params.numBonds > 0) { // read pair bonding parameters  : upper triangular matrix
                        auto rij = json_parser::get_vector<double>(loc["FENE"], "RMAX_DIVISION", params.numBonds);
                        int shift = 0;
                        int count = 0;
                        for (auto i : irange(0, params.numElems - 1)) {
                            shift += i + 1; //index shift to upper diagonal element
                            for (auto j : irange(i + 1, params.numElems)) {
                                auto im = i * params.numElems + j - shift;
                                params.divisionR2max(i, j) = params.divisionR2max(j, i) = SQ(rij[im] * units.Length);
                                count++;
                            }
                        }
                        run_assert(count == params.numBonds, "Incorrect number of bonds!");
                    }
                } else {
                    for (auto i : irange(0, params.numElems - 1)) {
                        for (auto j : irange(i + 1, params.numElems)) {
                            params.divisionR2max(i, j) = params.divisionR2max(j, i) = params.r2max(i, j);
                        }
                    }
                } // Division version of Rmax
            } else {
                params.DIV = false;
                // params.divisionPressure = 0.0;
                params.divisionR0 = params.divisionR2min = 0.0;
                params.divisionTau.mu = params.divisionTau.sigma = 0.0;
                params.migrationTau.mu = params.migrationTau.sigma = 0.0;
                params.divisionR2max = 0.0;
                params.divisionR2minRatio = 0;
                params.divisionSigmaBirthRatio = 0;
                params.divisionSigmaMin = 0;

            }
            if (params.DIV) SWITCH.set(OPT::CELLS_DIVIDING);
            if (params.divisionConstForce) SWITCH.set(OPT::DIV_CONST_FORCE);
            if (params.divisionSwelling) SWITCH.set(OPT::SWELLING_ON);
        }

        //Read apoptosis parameters (optional)
        {
            if (json_parser::exists(loc, "APOPTOSIS")) {
                if (json_parser::get<bool>(loc["APOPTOSIS"], "ENABLED")) {
                    auto aloc = json_parser::get_child(loc, "APOPTOSIS");
                    params.APO = true;
                    params.apopPressure = json_parser::get<double>(aloc, "APOP_PRESSURE") * units.Pressure;
                    params.apopDuration = json_parser::get<double>(aloc, "APOP_DURATION") * units.Time;
                    params.apopRate = json_parser::get<double>(aloc, "APOP_RATE") / units.Time;
									  params.apopAlpha = json_parser::get<double>(aloc, "APOP_ACTIVITY_ALPHA");
									  params.apopGamma = json_parser::get<double>(aloc, "APOP_GAMMA");
									  params.extrRate = json_parser::get<double>(aloc, "EXTR_RATE") / units.Time;
									  params.extrGamma = json_parser::get<double>(aloc, "EXTR_GAMMA");
                    if (params.APO) SWITCH.set(OPT::CELLS_DYING);
                }
            }
        }
    }// Species

    { // Set elem parameters
        colony.paramElems.resize(colony.getElemTypes());
        for (auto &cell : colony.paramCells) {
            for (auto i : irange(0, cell.numElems)) {
                auto type = cell.elemType[i];
                auto &elem = colony.paramElems[type];
                elem.type = type;
                elem.m = cell.m[i];
                elem.growthk0 = cell.growthk0[i];
                elem.growthk1 = cell.growthk1[i];
                elem.growthk2 = cell.growthk2[i];
                elem.zeta = cell.zeta[i];
                elem.zeta_p = cell.zeta_p[i];
                elem.zeta_p_beta = cell.zeta_p_beta[i];
                elem.ms = cell.ms[i];
                elem.zetaInv = cell.zetaInv[i];
                elem.sigmaCore = elem.sigmaCoreX = cell.sigmaCore[i];
                elem.sigmaWell = cell.sigmaWell[i];
                elem.epsilonCore = cell.epsilonCore[i];
                elem.epsilonWell = cell.epsilonWell[i];
            }
        }
    }
    std::cerr << "ok" << std::endl;
}

void readInteractionsLJ(const json &jiop, array2<double> &rc2) {
    const double r2HS = pow(2.0, 1.0 / 3.0);
    rc2 = r2HS; //Default to Lennard-Jones Hard-Sphere cutoff

    auto attraction = json_parser::get<string>(jiop["ATTRACTION"], "SELECT", {"ON", "OFF"});
    if (attraction == "OFF") {
        rc2 = r2HS;
    } else {
        auto loc = json_parser::get_child(jiop["ATTRACTION"], "ON");
        auto r2LJ = SQ(json_parser::get<double>(loc, "RCUT"));
        auto shift = json_parser::get<string>(loc, "SHIFT", {"FORCE", "POTENTIAL"});
        if (shift == "FORCE") SWITCH.set(OPT::FORCE_SHIFTED);
        run_assert(r2LJ >= r2HS, "Invalid LJ cutoff");

        auto dloc = json_parser::get_child(loc, "DOMAIN");
        auto select = json_parser::get<string>(dloc, "SELECT", {"ALL", "SPECIES", "ELEMENTS"});
        if (select == "ALL") {
            rc2 = r2LJ;
        } else if (select == "SPECIES") {
            auto pairs = json_parser::get<vector < vector < string>>>(dloc, "SPECIES");
            for (auto &pair : pairs) {
                run_assert(pair.size() == 2, "Wrong cell pair specification");
                int spec_i = colony.name2CellType(pair[0]);
                int spec_j = colony.name2CellType(pair[1]);

                auto &params_i = colony.paramCells[spec_i];
                auto &params_j = colony.paramCells[spec_j];

                for (auto ielem : params_i.elemType) {
                    for (auto jelem : params_j.elemType) {
                        rc2(ielem, jelem) = rc2(jelem, ielem) = r2LJ;
                    }
                }
            }
        } else if (select == "ELEMENTS") { // "ELEMENTS"
            auto pairs = json_parser::get<vector < vector < vector < string>>>>(dloc, "ELEMENTS");

            for (auto &pair : pairs) {
                run_assert(pair.size() == 2, "Wrong cell pair specificiation");
                run_assert(pair[0].size() == 2 && pair[1].size() == 2, "Wrong (cell,elem) specification");

                int spec_i = colony.name2CellType(pair[0][0]);
                int spec_j = colony.name2CellType(pair[1][0]);
                auto &params_i = colony.paramCells[spec_i];
                auto &params_j = colony.paramCells[spec_j];
                auto it = params_i.mapNames.find(pair[0][1]);
                auto jt = params_j.mapNames.find(pair[1][1]);
                run_assert(it != params_i.mapNames.end(),
                           "Unknown element name " + pair[0][1] + " for cell type " + pair[0][0]);
                run_assert(jt != params_j.mapNames.end(),
                           "Unknown element name " + pair[1][1] + " for cell type " + pair[1][0]);
                auto ielem = params_i.elemType[it->second];
                auto jelem = params_j.elemType[jt->second];
                rc2(ielem, jelem) = rc2(jelem, ielem) = r2LJ;
            }
        }
    }
}

void readInteractionsSC(const json &jiop, array2
    <double> &rc2) {
    // Nothing to specify...
    rc2 = 1.0;
}

void readInteractionsHZ(const json &jiop, array2<double> &rc2) {
    // Nothing to specify...
    rc2 = 1.0;
}
void readInteractionsEA(const json &jiop, array2<double> &rc2) {
	  // Nothing to specify...
	  rc2 = 1.0;
}
void readInteractionsWall(const json &jiop) { // DIMENSION WARNING
    auto names = json_parser::get<vector < string>>
    (jiop, "DEFINE");
    std::vector<double> ls;
    container.box.length(ls);
    if (names.size() > 0) SWITCH.set(OPT::WALLS_ON);
    for (auto i : irange(0ul, names.size())) {
        auto &name = names[i];
        auto loc = json_parser::get_child(jiop, name);
        auto type = json_parser::get<string>(loc, "TYPE", {"CIRCULAR", "SLAB", "H_SLAB", "P_SLAB"});
        auto epsilon = json_parser::get<double>(loc, "EPSILON");
        epsilon *= units.Energy;

        double rcut = Constants::SIXTH_ROOT_TWO;
        bool attractive = json_parser::get<bool>(loc, "ATTRACTIVE");
        if (attractive) {
            rcut = json_parser::get<double>(loc, "RCUT");
        }

        if (type == "CIRCULAR") {
            auto origin = json_parser::get_vector<double>(loc, "ORIGIN", DIM);
            auto radius = json_parser::get<double>(loc, "RADIUS");
            auto direction = (json_parser::get<string>(loc, "DIRECTION", {"POSITIVE", "NEGATIVE"})
                              == "POSITIVE" ? true : false);

            for (auto d : irange(0, DIM)) origin[d] *= units.Length;
            radius *= units.Length;
            for (auto d : irange(0, DIM))
                run_assert(origin[d] - radius >= 0.0 && origin[d] + radius <= ls[d], "invalid range for circ. wall");
            paramsWall.addWall(name, new CircularWall(origin.data(), radius, rcut, epsilon, direction));
        } else if (type == "SLAB") {
            auto origin = json_parser::get_vector<double>(loc, "ORIGIN", 2);
            auto direction = json_parser::get<string>(loc, "DIRECTION", {"X", "Y", "Z"});
            int dir;
            if (direction == "X") {
                dir = 0;
            } else if (direction == "Y") {
                dir = 1;
            } else {
                run_assert(DIM == 3, "ZSLAB only makes sense for 3D systems");
                dir = 2;
            }
            origin[0] *= units.Length;
            origin[1] *= units.Length;
            run_assert(origin[0] >= 0.0 && origin[0] <= ls[dir], "invalid range for lower slab wall");
            run_assert(origin[1] >= 0.0 && origin[1] <= ls[dir], "invalid range for upper slab wall");
            paramsWall.addWall(name + "0", new FlatWall(origin[0], dir, rcut, epsilon, true));  //lower wall
            paramsWall.addWall(name + "1", new FlatWall(origin[1], dir, rcut, epsilon, false)); //upper wall
        } else if (type == "P_SLAB") {
            auto origin = json_parser::get_vector<double>(loc, "ORIGIN", 2);
            auto direction = json_parser::get<string>(loc, "DIRECTION", {"X", "Y", "Z"});
            auto endpoints = json_parser::get_vector<double>(loc, "ENDPOINTS", 2);
            auto up_low = json_parser::get_vector<bool>(loc, "UPPER_LOWER", 2);

            int dir;
            if (direction == "X") {
                dir = 0;
            } else if (direction == "Y") {
                dir = 1;
            } else {
                run_assert(DIM == 3, "ZSLAB only makes sense for 3D systems");
                dir = 2;
            }
            origin[0] *= units.Length;
            origin[1] *= units.Length;
            run_assert(origin[0] >= 0.0 && origin[0] <= ls[dir], "invalid range for lower slab wall");
            run_assert(origin[1] >= 0.0 && origin[1] <= ls[dir], "invalid range for upper slab wall");

            if (up_low[0])
                paramsWall.addWall(name + "0",
                                   new PartFlatWall(origin[0], dir, endpoints[0], endpoints[1], rcut, epsilon,
                                                    true));  //lower wall
            if (up_low[1])
                paramsWall.addWall(name + "1",
                                   new PartFlatWall(origin[1], dir, endpoints[0], endpoints[1], rcut, epsilon,
                                                    false)); //upper wall
        } else if (type == "H_SLAB") {
            auto origin = json_parser::get<double>(loc, "ORIGIN");
            auto direction = json_parser::get<string>(loc, "DIRECTION", {"X", "Y", "Z"});
            auto epsilon = json_parser::get<double>(loc, "EPSILON");
            rcut = json_parser::get<double>(loc, "RCUT");
            int dir;
            if (direction == "X") {
                dir = 0;
            } else if (direction == "Y") {
                dir = 1;
            } else {
                run_assert(DIM == 3, "ZSLAB only makes sense for 3D systems");
                dir = 2;
            }
            origin *= units.Length;
            run_assert(origin >= 0.0 && origin <= ls[dir], "invalid range for slab wall");
            paramsWall.addWall(name + "0", new HarmonicWall(origin, dir, rcut, epsilon, true));  //lower wall
        }
    }// over walls
}

void readInteractions(const json &jiop) {
    {
        std::cerr << "# Reading Pair Interactions ...";
        int numElems = colony.getElemTypes();
        array2<double> rc2(numElems, numElems);
        rc2 = 0.0;

        auto loc = json_parser::get_child(jiop, "PAIRFORCES");
        if (!SWITCH[OPT::FORCE_GHOSTS]) {
            if (SWITCH[OPT::FORCE_SOFTCORE]) {
                readInteractionsSC(loc["SC"], rc2);
                paramsForce = new SCParams;
            } else if (SWITCH[OPT::FORCE_HERTZIAN]) {
                readInteractionsHZ(loc["HZ"], rc2);
                paramsForce = new HZParams;
            } else if (SWITCH[OPT::FORCE_EA]) {
							readInteractionsEA(loc["EA"], rc2);
							paramsForce = new EAParams;
						}else {
            	readInteractionsLJ(loc["LJ"], rc2);
            	paramsForce = new LJParams;
            	}
            paramsForce->init(colony.paramElems, rc2);  //initialize pair-parameters
        }
        std::cerr << "ok" << std::endl;
    }
    {
        std::cerr << "# Reading Wall Interactions ...";
        readInteractionsWall(jiop["WALLS"]);
        std::cerr << "ok" << std::endl;
    }
}

/*!
 Read MD parameters
 */
void readMD(const json &jiop) {
    DIM = json_parser::get<int>(jiop, "DIM", {2, 3});
    if (DIM == 2) {
        ACTIVE_NOISE_DIM = 2; // radial + angular degrees of freedom
    } else {
        ACTIVE_NOISE_DIM = 3; // radial + 2 angular
    }

    std::cerr << "# Reading md parameters ...";
    {
        paramsMD.gts = json_parser::get<double>(jiop, "GTS");
        paramsMD.frames = json_parser::get<double>(jiop, "FRAMES");
    }
    {
        auto lengths = json_parser::get_vector<double>(jiop, "BOX", DIM);
        for (auto &l : lengths) l *= units.Length;
        auto pbc = json_parser::get_vector<bool>(jiop, "PBC", DIM);
        paramsMD.drain = json_parser::get<bool>(jiop, "DRAIN");
        paramsMD.tau_ve = json_parser::get<double>(jiop, "TAU_VE");//*units.Time
        container.box.init(DIM, lengths, pbc);
    }
    {
        auto loc = json_parser::get_child(jiop, "INTEGRATOR");
        auto integrator = json_parser::get<string>(loc, "SELECT",
                                                   {"EULER", "STOCHASTIC_EULER", "STOCHASTIC_HEUN", "DPD"});
        if (integrator == "EULER") {
            mdStep = Euler;
            SWITCH.set(OPT::INTEGRATOR_EULER);
        } else if (integrator == "STOCHASTIC_EULER") {
            SWITCH.set({OPT::NOISE_ON});
            mdStep = Euler;
            SWITCH.set(OPT::INTEGRATOR_STOCHASTIC_EULER);
        } else if (integrator == "STOCHASTIC_HEUN") {
            SWITCH.set({OPT::NOISE_ON, OPT::NOISE_ACTIVE});
            mdStep = StochasticHeun;
            SWITCH.set(OPT::INTEGRATOR_STOCHASTIC_HEUN);
        } else if (integrator == "DPD") {
            mdStep = DPD;
            SWITCH.set(OPT::INTEGRATOR_DPD);
            auto loc2 = json_parser::get_child(loc, "DPD");
            paramsMD.dissamcel = json_parser::get<double>(loc2, "DISSAMCEL");
            paramsMD.disdifcel = json_parser::get<double>(loc2, "DISDIFCEL");
            paramsMD.disbg = json_parser::get<double>(loc2, "DISBG");
        } else {
            exit(-1);
        }
        run_assert(!(SWITCH[OPT::NOISE_ACTIVE] && DIM == 3), "Active noise not yet implemented for 3D systems");
    }
    {
        paramsMD.temp = json_parser::get<double>(jiop, "TEMP");
        paramsMD.pi = json_parser::get<double>(jiop, "OSMOTIC_PRESSURE");
        paramsMD.pi *= units.Pressure;
        //paramsMD.KBT  = (Constants::KB * paramsMD.temp);
        // Now I just directly choose kBT.
        paramsMD.kbt = json_parser::get<double>(jiop, "KBT");
        paramsMD.kbt *= units.Kbtn;

    }
    std::cerr << "ok" << std::endl;
}

void readRunTime(const json &jiop) {
    std::cerr << "# Reading run time parameters ...";
    { // link list parameters
        if (json_parser::get<bool>(jiop["LINKLIST"], "ENABLED") && !SWITCH[OPT::FORCE_GHOSTS]) {
            SWITCH.set(OPT::LINKLIST_ON);
            std::cerr << " reading linkList parameters...";
            if (json_parser::get<bool>(jiop["LINKLIST"], "STENCIL")) SWITCH.set(OPT::LINKLIST_STENCIL);
            if (json_parser::get<bool>(jiop["LINKLIST"], "SORTED")) SWITCH.set(OPT::LINKLIST_SORTED);
            // The link list is generated later, in CIL.hpp: initializeLinkList,
            // when we have all information about the cells
        }
    }
    { // random number generator
        std::cerr << " Init random number generator ...";
        auto loc = json_parser::get_child(jiop, "RNG");
        int seed = json_parser::get<int>(loc, "SEED");
        if (seed <= 0) {
            Random::init(time(NULL));
            std::cerr << " Init seed is random...";
        } else {
            Random::init(seed);
            std::cerr << " Init seed is " << seed << " ...";
        }
    }
    std::cerr << "ok" << std::endl;
}

/*!
 Read Initial configuration
 */
void readInit(const json &jiop) {
    std::cerr << "# Reading initial configuration ...";
    vector<double> ls;
    container.box.length(ls);
    for (auto cellType : irange(0, colony.getCellTypes())) {
        auto &name = colony.nameCells[cellType];
        auto &params = colony.paramCells[cellType];
        auto &pos = colony.pos[cellType];
        auto &vel = colony.vel[cellType];
        auto &state = colony.cellState[cellType];
        auto &theta = colony.cellTTheta[cellType];
        auto &rho = colony.cellTRho[cellType];
        auto &sigmaCore = colony.sigmaCore[cellType];
        auto &sigmaCoreT = colony.sigmaCoreT[cellType];
        auto &sigmaTheta = colony.sigmaTheta[cellType];
        // Initialize cell positions
        auto &loc = json_parser::get_child(jiop, name);
        auto type = json_parser::get<string>(loc, "SELECT", {"JSON", "H5", "RANDOM"});
        if (type == "H5") {
            std::cout << " attempt reading from h5 file ...";
            auto hloc = json_parser::get_child(loc, "H5");

            SimIO::SimIO fp;
            std::vector<uint> hdims;

            // TODO: Improve this by -- at least -- better labeling the items in the json file.
            type = json_parser::get<string>(hloc, "SELECT", {"ALL", "LOCATION"});
            
            if (type == "LOCATION") {
                auto restart = json_parser::get_vector<string>(hloc, "LOCATION", 2);   // file and location.
                fp.open_file(restart[0].c_str(), "r");
                fp.read_data_ndims(restart[1].c_str(), hdims);
                auto numData = std::accumulate(hdims.begin(), hdims.end(), 1, std::multiplies<uint>());
                auto numCells = numData / (params.numElems * DIM);
                run_assert(numCells * params.numElems * DIM == numData, "Inconsistent H5 dataset dimensions");

                vector<int> cellRange{0, numCells}; // (start_cell, end_cell)
                if (json_parser::get_size(hloc, "RANGE") > 0) {
                    cellRange = json_parser::get_vector<int>(hloc, "RANGE", 2);
                    for (auto &ii : cellRange) if (ii < 0) ii += numCells;
                }
                for (auto &ii : cellRange) run_assert(ii >= 0 && ii <= numCells, "Range out of bounds");
                run_assert(cellRange[0] < cellRange[1], "Range is empty");

                auto space = SimIO::Space::create(numCells, params.numElems, DIM); // nCells | elem | dim
                array3<double> auxPos(numCells, params.numElems, DIM);
                fp.read_data(restart[1].c_str(), space, auxPos());

                {
                    auto targetCells = cellRange[1] - cellRange[0];
                    array3<double> cpyPos(targetCells, params.numElems, DIM,
                                          auxPos() + cellRange[0] * params.numElems * DIM);
                    colony.setNumCells(cellType, targetCells);
                    std::copy(cpyPos.begin(), cpyPos.end(), pos());
                }
                SimIO::Space::close(space);
                fp.close_file();
            } else if (type == "ALL") {
                auto restart = json_parser::get_vector<string>(hloc, "ALL", 7);   // file and location within of the data for position, velocities, state, theta, rho, sigmaCore
                fp.open_file(restart[0].c_str(), "r");
                fp.read_data_ndims(restart[1].c_str(), hdims);
                auto numData = std::accumulate(hdims.begin(), hdims.end(), 1, std::multiplies<uint>());
                auto numCells = numData / (params.numElems * DIM);
                run_assert(numCells * params.numElems * DIM == numData, "Inconsistent H5 dataset dimensions");

                vector<int> cellRange{0, numCells}; // (start_cell, end_cell)
                if (json_parser::get_size(hloc, "RANGE") > 0) {
                    cellRange = json_parser::get_vector<int>(hloc, "RANGE", 2);
                    for (auto &ii : cellRange) if (ii < 0) ii += numCells;
                }
                for (auto &ii : cellRange) run_assert(ii >= 0 && ii <= numCells, "Range out of bounds");
                run_assert(cellRange[0] < cellRange[1], "Range is empty");

                std::cerr << " creating spaces...";
                auto space1 = SimIO::Space::create(numCells, params.numElems, DIM); //POS, nCells | elem | dim
                auto space2 = SimIO::Space::create(numCells, params.numElems, DIM); //VEL, nCells | elem | dim
                auto space3 = SimIO::Space::create(numCells); // STATE, nCells
                auto space4 = SimIO::Space::create(numCells); // THETA, nCells
                auto space5 = SimIO::Space::create(numCells); // RHO, nCells
                auto space6 = SimIO::Space::create(numCells, params.numElems); // Diameters, nCells | elem

                std::cerr << " creating aux...";
                array3<double> auxPos(numCells, params.numElems, DIM);
                array3<double> auxVel(numCells, params.numElems, DIM);
                array1<int> auxSta(numCells);
                array1<double> auxTheta(numCells);
                array1<double> auxRho(numCells);
                array2<double> auxSigmaCore(numCells, params.numElems);
							  array2<double> auxSigmaCoreT(numCells, params.numElems);

                std::cerr << " actually read the data..." ;
                std::cerr << "Pos, ";
                fp.read_data(restart[1].c_str(), space1, auxPos());
                std::cerr << "Vel, ";
                fp.read_data(restart[2].c_str(), space2, auxVel());
                std::cerr << "State, ";
                fp.read_data(restart[3].c_str(), space3, auxSta());
                std::cerr << "Theta, ";
                fp.read_data(restart[4].c_str(), space4, auxTheta());
                std::cerr << "Rho, " ;
                fp.read_data(restart[5].c_str(), space5, auxRho());
                std::cerr << "SigmaCore, ";
                fp.read_data(restart[6].c_str(), space6, auxSigmaCore());
                fp.read_data(restart[6].c_str(), space6, auxSigmaCoreT());
                std::cerr << " done...";
                {
                    std::cerr << " copy the data..." << std::endl;
                    auto targetCells = cellRange[1] - cellRange[0];

                    array3<double> cpyPos(targetCells, params.numElems, DIM,
                                          auxPos() + cellRange[0] * params.numElems * DIM);
                    array3<double> cpyVel(targetCells, params.numElems, DIM,
                                          auxVel() + cellRange[0] * params.numElems * DIM);
                    array1<int> cpySta(targetCells, auxSta() + cellRange[0]);
                    array1<double> cpyTheta(targetCells, auxTheta() + cellRange[0]);
                    array1<double> cpyRho(targetCells, auxRho() + cellRange[0]);
                    array2<double> cpySigmaCore(targetCells, params.numElems,
                                                auxSigmaCore() + cellRange[0] * params.numElems);
                    array2<double> cpySigmaCoreT(targetCells, params.numElems,
                                                auxSigmaCoreT() + cellRange[0] * params.numElems);
                    colony.setNumCells(cellType, targetCells);

                    std::copy(cpyPos.begin(), cpyPos.end(), pos());
                    std::copy(cpyVel.begin(), cpyVel.end(), vel());
                    std::copy(cpySta.begin(), cpySta.end(), state());
                    std::copy(cpyTheta.begin(), cpyTheta.end(), theta());
                    std::copy(cpyRho.begin(), cpyRho.end(), rho());
                    std::copy(cpySigmaCore.begin(), cpySigmaCore.end(), sigmaCore());
                    std::copy(cpySigmaCoreT.begin(), cpySigmaCoreT.end(), sigmaCoreT());
                }
                SimIO::Space::close(space1);
                SimIO::Space::close(space2);
                SimIO::Space::close(space3);
                SimIO::Space::close(space4);
                SimIO::Space::close(space5);
                SimIO::Space::close(space6);
                fp.close_file();
            }
        }
        else if (type == "JSON") { // JSON
            auto numCells = json_parser::get_size(loc, "JSON") / params.numElems;
            colony.setNumCells(cellType, numCells);
            json_parser::load_data(loc, "JSON", pos(), numCells * params.numElems, DIM);
            for (auto i : irange(0ul, numCells))
                for (auto j : irange(0, params.numElems))
                    for (auto d : irange(0, DIM))
                        pos(i, j, d) *= units.Length;
        }
        else if (type == "RANDOM") {
            auto numCells = json_parser::get<unsigned int>(loc["RANDOM"], "NUMBER");
            colony.setNumCells(cellType, numCells);
            const array3<double> &pos = colony.pos[cellType];
            for (auto i : irange(0u, numCells)) {
                const auto &Ri = pos[i];
                for (auto j : irange(0, params.numElems))
                    for (auto d : irange(0, DIM))
                        Ri(j, d) = Random::uniform0x(ls[d]);
                for (auto j : irange(1, params.numElems)) {
                    auto radius = Random::normal(params.rss(0, j), 0.25 * params.rss(0, j));
                    auto theta = Random::uniform() * Constants::PI;
                    auto sinphi = (DIM == 2 ? 1.0 : Random::uniform() * Constants::PI);
                    double dir[MAX_DIM] = {sinphi * cos(theta), sinphi * sin(theta), sqrt(1.0 - SQ(sinphi))};
                    container.box.updatePos(Ri[j], dir, radius);
                }
            }
        }

    }
    std::cerr << "ok" << std::endl;
}

void readOutput(const json &jiop) {
    std::cerr << "# Reading output parameters ...";
    paramsOut.dirName = json_parser::get<string>(jiop, "DIR");
    paramsOut.prjName = json_parser::get<string>(jiop, "NAME");
    auto dumps = json_parser::get<vector < string>>
    (jiop, "FORMAT");
    for (auto &dump : dumps) {
        if (dump == "SILO") {
#ifdef WITHSILO
            SWITCH.set({OPT::OUTPUT_SILO, OPT::OUTPUT_DUMP});
#else
            std::cerr << "Compiled without silo support!" << std::endl;
            exit(1);
#endif
        } else if (dump == "GSD") {
            SWITCH.set({OPT::OUTPUT_GSD, OPT::OUTPUT_DUMP});
        } else if (dump == "H5") {
            SWITCH.set({OPT::OUTPUT_H5, OPT::OUTPUT_DUMP});
        } else if (dump == "LAMMPS") {
            SWITCH.set({OPT::OUTPUT_LAMMPS, OPT::OUTPUT_DUMP});
        } else {
            std::cerr << "Unknown data format: " << dump << std::endl;
            exit(1);
        }
    }
    paramsOut.energy_record = json_parser::get<bool>(jiop, "ENERGY_RECORD");
    paramsOut.div_app_record = json_parser::get<bool>(jiop, "DIV_APP_RECORD");
    std::cerr << "ok" << std::endl;
}
