#ifndef CIL_CUDA_HPP
#define CIL_CUDA_HPP

#include "Output.hpp"
#include "OutputWriter.hpp"
#include "Input.hpp"
#include "MD.hpp"
#include "Random.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h> // CUDA helper functions
#include <curand_kernel.h>

/*! \file CIL_CUDA.hpp
 \brief Initialization and finalization of the simulation with CUDA
 */

 void initializeColony(const json &jiop){
    // Init forces functions
    for(auto cellType : irange(0, colony.getCellTypes())){
      const auto &params = colony.paramCells[cellType];
      colony.migrationFunc[cellType] = NULL;
      colony.divisionFunc[cellType]  = NULL;
      colony.divisionFail[cellType]  = NULL;
      colony.divisionCheck[cellType] = NULL;
      // Set intra-cellular force function
      if(params.numElems == 2){
        colony.migrationFunc[cellType]   = binaryCellForce;
        if(params.DIV){
          colony.divisionFunc[cellType]  = binaryCellDivision;
          colony.divisionFail[cellType]  = binaryCellDivisionFail;
          colony.divisionCheck[cellType] = binaryCellDivisionCriteria;
        }
        // Set force function for single-particle cells. Also dependent on cell shape. By default, it's circle, shape = 0.
      }else if(params.numElems == 1){
          colony.migrationFunc[cellType] = singleCellForce;
          if(params.DIV){
              // if(params.shape){
              //     colony.divisionFunc[cellType]  = singleRodDivision;
              //     colony.divisionCheck[cellType] = singleRodDivisionCriteria;
              // }else{
                  colony.divisionFunc[cellType]  = singleCellDivision;
                  colony.divisionCheck[cellType] = singleCellDivisionCriteria;
              // }
          }
      }
    }
  
    // Init configuration
    readInit(jiop);
    colony.synchronize();
  
    // Confirm that positions are valid
    for(auto cellType : irange(0, colony.getCellTypes())){
      auto nCells = colony.getNumCells(cellType);
      auto nElems = colony.paramCells[cellType].numElems;
      auto &pos   = colony.pos[cellType];
      for(auto i : irange(0,nCells))
        for(auto j : irange(0,nElems)) 
        { 
          if (!container.box.isValidPos(pos[i][j])){
            std::cerr << "Input Positions Out of Bounds!" << std::endl;
            exit(1);
          }
        }
    }
  }
  
  void initializeLinkList(const json &jiop) {
    // Configure Linked List if enabled
    if (SWITCH[OPT::LINKLIST_ON]) {
      std::cerr << "# Configuring linkList ...";
      if (!SWITCH[OPT::LINKLIST_STENCIL]) {
        // Standard linked list (NOT stencil link list):
        // Provide the linked list with (1) the cutoff
        // radius of the interaction and (2) a size criterion for the volume
        // regulation, and (3) the largest extent that the cells can have so that
        // the linked list will correctly include all cells whose elements could
        // possibly interact in neighboring linked list cells. The cell extent is
        // the farthest possible distance from the cell's center of mass that an
        // element can have.
        // TODO Reconsider the above statement as r2max refers to the
        // square of the maximum extension of FENE springs between
        // cell elements, I believe
  
        // for the case of static diameters, the element size is fully determined
        // by the following
        double rc2 = 0.0;  // maximum square of cutoff radius times element diameter
        auto &elems = colony.paramElems;
        auto types = elems.size();
        for (auto i : irange(0lu, types)) {
          for (auto j : irange(i, types)) {
            rc2 = MAX(rc2, paramsForce->getRc2(elems[i], elems[j]));
          }
        }
        // TODO Reconsider this paragraph, seems non-sensical
        CellParams &params = colony.paramCells[types];
        if (params.DIV) rc2 = 2 * rc2;
  
        // If the division growth mechanism is enabled, cells can change in size.
        // If that's the case, we need to make an estimate of their maximum
        // squared interaction diameter. We calculate the maximum size in terms of
        // V1. We know that cells can be larger than that, e.g. due to negative
        // pressures.
        // TODO This should either be calculated precisely here, or cell sizes
        // should be checked periodically and if exceeding V1, the link list
        // should be resized. For now, we assume that the cells won't exceed the
        // heuristic below. Perhaps this heuristic factor should be controllable
        // from the json.
        double heuristic_volume_factor = 1.5;
        double rc2_dynamic = 0.0;
        if (SWITCH[OPT::CELLS_DIVIDING]){
          std::cerr << "for linklist cell sizing, assume that cells are dynamically sized up to a volume/area of " << heuristic_volume_factor << "*V1, heuristically accounting for slightly larger volumes due to motility. Be sure to check that this is self-consistent with the simulation results ...";
          run_assert(SWITCH[OPT::FORCE_HERTZIAN],
                     "ERROR: Calculation of the largest interaction diameter for growing cells only implemented for Hertzian Contact Mechanics so far, because we rely on rcut=1 and there not being any special interaction rules.\n");
          for (auto cellType : irange(0, colony.getCellTypes())) {
            auto &params = colony.paramCells[cellType];
            run_assert(params.numElems == 1,
                       "ERROR: Currently, the division code only works for "
                       "one-element cells anymore.\n");
            rc2_dynamic = MAX(rc2_dynamic, 4. * params.v_1 / Constants::PI);
          }
          rc2_dynamic *= heuristic_volume_factor;
        }
  
        double r2max = 0.0;  // square of the largest cell extent
        for (auto ispec : irange(0, colony.getCellTypes())) {
          auto &paramCells = colony.paramCells[ispec];
          for (auto i : irange(0, paramCells.numElems))
            for (auto j : irange(0, paramCells.numElems))
              r2max = MAX(r2max, paramCells.r2max(i, j));
        }
  
        double linkListCellWidth =
            (MAX(sqrt(rc2), sqrt(rc2_dynamic)) + sqrt(r2max));
        container.linkList.init(container.box, linkListCellWidth);
  
      } else {             // LinkList_Stencil: This here follows Jintao's idea
                           // for how this should go, probably false
        double rc2 = 100;  // square of the cutoff radius
        auto &elems = colony.paramElems;
        auto types = elems.size();
  
        for (auto i : irange(0lu, types))
          for (auto j : irange(i, types))
            rc2 = MIN(rc2, paramsForce->getRc2(elems[i], elems[j]));
  
        double r2max = 100;  // square of the largest cell extent
        for (auto ispec : irange(0, colony.getCellTypes())) {
          auto &paramCells = colony.paramCells[ispec];
          for (auto i : irange(0, paramCells.numElems))
            for (auto j : irange(0, paramCells.numElems))
              r2max = MIN(r2max, paramCells.r2max(i, j));
        }
        double linkListCellWidth = 2 * (sqrt(rc2) + sqrt(r2max));
        container.linkList.init(container.box, linkListCellWidth);
      }
      if (container.linkList.numCells == 1) {
        std::cerr << " Disabling link list because container size only allows for one link list cell ...";
        SWITCH.set(OPT::LINKLIST_ON, false);
        SWITCH.set(OPT::LINKLIST_STENCIL, false);
      }
      std::cerr << "ok" << std::endl;
    }
  }
  
  void initializeTimeStep(const json &jiop){
    bool automatic_dt = false;
    double tau_intra, tau_inter, tau_scale;
    tau_intra = tau_inter = Constants::MAX_MP;
    tau_scale = 1.0;
  
    // migration & fene
    for(auto cellType : irange(0, colony.getCellTypes())){
      auto &params = colony.paramCells[cellType];
      auto elems  = params.numElems;
  
      for(auto i : irange(0, elems)){
        for(auto j : irange(i+1, elems)){
      auto lmin = params.rss(i,j);
      auto fmax = MAX(params.kappa(i,j), params.m(i)) * sqrt(params.r2max(i,j));
      if(fmax > 0.0) tau_intra = MIN(tau_intra, params.zeta(i) * lmin / fmax);
        }
      }
    }
  
    // interparticle interactions
    if(!SWITCH[OPT::FORCE_GHOSTS]){
      int elemTypes = colony.paramElems.size();
      for(auto i : irange(0, elemTypes)){
        auto &ei = colony.paramElems[i];
        for(auto j : irange(i, elemTypes)){
      auto &ej = colony.paramElems[j];
      tau_inter = MIN(tau_inter, paramsForce->getTimeScale(ei, ej)*MIN(ei.zeta, ej.zeta));
        }
      }
    }
  
    {// read dt
      if(json_parser::get<string>(jiop,"SELECT", {"AUTOSCALE","MANUAL"}) == "AUTOSCALE"){
        tau_scale    = json_parser::get<double>(jiop, "AUTOSCALE");
        paramsMD.dt  = MIN(tau_intra, tau_inter) * tau_scale;
        automatic_dt = true;
      }else{
        paramsMD.dt  = json_parser::get<double>(jiop, "MANUAL")*units.Time;
      }
      run_assert(paramsMD.dt > 0.0, "dt <= 0");
      paramsMD.hdt = 0.5*paramsMD.dt;
    }
    {// include dt factor in noise amplitudes
      auto isqrtdt = 1.0 / sqrt(paramsMD.dt);
      for(auto cellType : irange(0, colony.getCellTypes())){
        auto &params = colony.paramCells[cellType];
        params.dNoise *= isqrtdt;
      }
    }
  
    // include dt factor in cell cycle timescales
    for(auto cellType : irange(0, colony.getCellTypes())){
      auto &params = colony.paramCells[cellType];
      if (params.DIV){
        params.cycleTauR  /= paramsMD.dt;
        params.cycleTauV  /= paramsMD.dt;
        params.cycleTauP  /= paramsMD.dt;
        params.cyclePShiftTauInv *= paramsMD.dt;
      }
    }
    
    // include dt factor in apoptosis rate
    for(auto cellType : irange(0, colony.getCellTypes())){
      auto &params = colony.paramCells[cellType];
      if (params.APO){
        params.apopRate *= paramsMD.dt;
        //run_assert(params.apopRate > 0.0, "Apoptosis rate per time step <= 0");
        run_assert(params.apopRate < 1.0, "Apoptosis rate per time step >= 1");
      }
    }
    
    {
      using std::cerr;
      using std::endl;
      cerr << fmtTitle % "Time Step" << endl;
      cerr << fmtHead  % "DT_AUTO?"; cerr << fmtStr % (automatic_dt ? "YES" : "NO") << endl;
      cerr << fmtHead  % "DT_INTRA" << fmtDbl % (tau_intra) << endl;
      cerr << fmtHead  % "DT_INTER" << fmtDbl % (tau_inter) << endl;
      cerr << fmtHead  % "DT_SCALE" << fmtDbl % (tau_scale) << endl;
      cerr << fmtHead  % "DT_SIMULATION" << fmtDbl % (paramsMD.dt) << endl;
      cerr << "#" << endl;
    }
  }
  
  // Initialize and randomize the cell states
  void initializeCellStatus(){
    std::cerr << "# Initializing the cell states ...";
    for(auto cellType : irange(0, colony.getCellTypes())){
      auto &params  = colony.paramCells[cellType];
      auto &state   = colony.cellState[cellType];
      auto &t0      = colony.cellT0[cellType];
      auto &tswitch = colony.cellTSwitch[cellType];
      auto &tpressure = colony.cellTPressure[cellType];
      auto &angVel  = colony.cellAngVel[cellType];
      // auto &angel   = colony.cellAngel[cellType];
      auto &theta   = colony.cellTTheta[cellType];
      auto &divPair = colony.divisionPair[cellType];
      auto &rho     = colony.cellTRho[cellType];
      t0 = 0;
      if(colony.migrationFunc[cellType] != NULL){
        // Set up the cell states randomly while roughly reflecting the steady state distribution of cell states.
        double probabilityOfMigrationState = params.migrationTau.mu/(params.migrationTau.mu + params.divisionTau.mu);
        auto nCells = colony.getNumCells(cellType);
        tswitch = -1;
        if(SWITCH[OPT::CELLS_DIVIDING]){
          for(auto i : irange(0, nCells)){
            // Initialise the cell state and its duration
            tpressure[i] = 0;
            int timeSpan = 0;
            // If the state of cells has already been specified during input, the state will remain same.
            // If not, it will be specified follow the steady state distribution of cell states.
                      if (params.divisionConstForce){
                          state[i] = (state[i]==2? CellState::Crawling : state[i]);
                          timeSpan = randomTimeSpan(params.migrationTau);
                      }
            else if (Random::uniform0x(1.0) <= probabilityOfMigrationState) {
              state[i] = (state[i]==2? CellState::Crawling : state[i]);
              timeSpan = randomTimeSpan(params.migrationTau);
            }else{
                state[i] = (state[i]==2? CellState::Dividing : state[i]);
              timeSpan = randomTimeSpan(params.divisionTau);
            }
            // Mimic a system in which the cell had entered its state before t0.
            // angel[i] = 0;
            angVel[i] = 1.0/timeSpan;
  
            theta[i] = Random::uniform0x(1);
            divPair[i] = -1;
            rho[i] = 1;
  
            timeSpan -= Nint(Random::uniform0x((double)timeSpan)) - 1;
            tswitch[i] = t0[i] + timeSpan;
  
            const double maxSigmaCore = params.sigmaCore.Max();
            const double sigmaMax = params.sigmaMax.Max();
  
            // auto rVel = angVel[i]*(sigmaMax - maxSigmaCore);
          }
        }
      }else{
        state   = CellState::Undefined;
        tswitch = -1;
      }
  
      // Set cell diameters
  
      auto &sigmaCore = colony.sigmaCore[cellType];
      auto &sigmaTheta = colony.sigmaTheta[cellType];
      auto &sigmaCoreT= colony.sigmaCoreT[cellType];
      auto nCells = colony.getNumCells(cellType);
      for(auto i : irange(0, nCells)) {
        for(auto j : irange(0, params.numElems)) {
  //          if (sigmaCore(i, j) != NULL) continue;
  //          else{
                sigmaCore(i, j) = params.sigmaCore(j);
                sigmaTheta(i,j) = params.sigmaTheta(j);
                sigmaCoreT(i,j) = params.sigmaCore(j);
  //          }
        }
      }
    }
    std::cerr << "ok" << std::endl;
  }
  
  void initialize(const json &jiop){
    using std::cerr;
    using std::endl;
  
    SWITCH.reset();
    units.init(json_parser::get<double>(jiop["UNITS"], "LENGTH"),
           json_parser::get<double>(jiop["UNITS"], "MOTILITY"),
           json_parser::get<double>(jiop["UNITS"], "FRICTION"));
    {
      string type = json_parser::get<string>(jiop["INTERACTIONS"]["PAIRFORCES"], "SELECT",
        {"LJ", "SC", "HZ", "EA","GHOSTS"}) ;
      if(type == "SC"){
        SWITCH.set(OPT::FORCE_SOFTCORE);
        std::cerr << "# Force type = Soft-Core"     << std::endl;
      }else if(type == "LJ"){
        std::cerr << "# Force type = Lennard-Jones" << std::endl;
      }else if(type == "HZ"){
        SWITCH.set(OPT::FORCE_HERTZIAN);
        std::cerr << "# Force type = Hertzian"     << std::endl;
      }else if(type == "GHOSTS"){
          SWITCH.set(OPT::FORCE_GHOSTS);
      }else if(type == "EA"){
              SWITCH.set(OPT::FORCE_EA);
      }
    }
  
    //Read system parameters
    try{
      readMD(jiop["MD"]);
      readSpecs(jiop["SPECIES"]);
      readInteractions(jiop["INTERACTIONS"]);
      readRunTime(jiop["RUNTIME"]);
      readOutput(jiop["OUTPUT"]);
    }catch(std::exception& e){
      std::cerr << e.what() << std::endl;
      exit(1);
    }
  
    //Initalize system
    try{
      initializeColony(jiop["INIT"]); // can we eliminitate this one?
      initializeTimeStep(jiop["MD"]["DT"]); // must run before initializeCellStatus()
      initializeCellStatus();
      initializeColony(jiop["INIT"]);
      initializeLinkList(jiop["INIT"]);
      initializeOutput();
    }catch(std::exception& e){
      std::cout << e.what() << std::endl;
      exit(1);
    }
  
    cerr << units     << endl;
    cerr << paramsOut << endl;
    cerr << paramsMD  << endl;
    cerr << paramsWall<< endl;
    cerr << container << endl;
    cerr << colony    << endl;
    /*if(SWITCH[OPT::FORCE_SOFTCORE]){
    }else if(!SWITCH[OPT::FORCE_GHOSTS]){// defaul lj
      cerr << dynamic_cast<LJParams&>(*paramsForce) << endl;
      }*/
  }
  void finalize(){
    finalizeOutput();
  }
  
  #endif
    