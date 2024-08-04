#include "Output.hpp"

// Output Unit Parameters
std::ostream & operator << (std::ostream &s, const Units &u){
  using std::endl;
  s << fmtTitle % "Simulation Units" << endl;
  s << fmtHead % "[Length]"       << fmtDbl % u.Length   << endl;
  s << fmtHead % "[Motility]"     << fmtDbl % u.Motility << endl;
  s << fmtHead % "[Friction]"     << fmtDbl % u.Friction << endl;
  s << fmtHead % "[Energy]"       << fmtDbl % u.Energy   << endl;
  s << fmtHead % "[Force]"        << fmtDbl % u.Force    << endl;
  s << fmtHead % "[Time]"         << fmtDbl % u.Time     << endl;
  s << fmtHead % "[Pressure]"     << fmtDbl % u.Pressure << endl;
  s << fmtHead % "[Density]"      << fmtDbl % u.Density  << endl;
  s << "#";
  return s;
}

// Output MD parameters
std::ostream & operator << (std::ostream &s, const MDParams &p){
  using std::endl;
  s << fmtTitle % "MD Parameters" << endl;
  s << fmtDataDbl % "dt"          % (p.dt) << endl;
  s << fmtDataDbl % "Frames"      % p.frames << endl;
  s << fmtDataInt % "GTS"         % p.gts << endl;
  s << fmtDataInt % "Total Steps" % (p.frames * p.gts) << endl;
  s << "#";
  return s;
}

std::ostream& operator << (std::ostream& s, const Container &c){
  using std::endl;
  double ls[MAX_DIM];
  bool   pbc[MAX_DIM];
  c.box.length(ls);
  c.box.isPeriodic(pbc);
  s << fmtTitle % "Box Parameters" << endl;
  s << fmtHead  % "Dim"; s << fmtInt % DIM; s << endl;

  s << fmtHead  % "Size [Red. Units]";
  for(auto d : irange(0, DIM)) s << fmtDbl % (ls[d] / units.Length);
  s << endl;

  s << fmtHead  % "Size [Sim. Units]";
  for(auto d : irange(0, DIM)) s << fmtDbl % ls[d];
  s << endl;

  s << fmtHead  % "PBC" ;
  for(auto d : irange(0, DIM)) s << fmtStr % (pbc[d] ? "YES" : "NO");
  s << endl;

  s << fmtTitle % "Link List" << endl;
  if(SWITCH[OPT::LINKLIST_ON]){
    std::vector<int> ns;
    c.linkList.link->get_ns(ns);
    s << fmtDataStr % "Enabled" % "YES"  << endl;;
    s << fmtDataStr % "Sorted " % (SWITCH[OPT::LINKLIST_SORTED] ? "YES" : "NO") << endl;
    s << fmtDataInt % "Number Cells" % c.linkList.numCells << endl;
    s << fmtDataInt % "Number Neighbors" % c.linkList.numNeighbors << endl;
    s << fmtSub % "Ns : ";
    for(auto d : irange(0, DIM)) s << fmtInt % ns[d];
  }else{
    s << fmtDataStr % "Enabled" % "NO";
  }
  return s;
}
std::ostream& operator << (std::ostream& s, const WallParams &p){
  using std::endl;
  if(SWITCH[OPT::WALLS_ON]){
    s << fmtTitle % "Walls" << endl;
    for(auto i : irange(0, p.size())){
      s << fmtSub % p.names[i] << endl;
      (p.walls[i])->dump(s);
    }
  }
  s << '#';
  return s;
}


// Output LJ parameters
/*std::ostream & operator << (std::ostream &s, const LJParams &p){
  using std::endl;
  array2<double> dmy(p.epsilon.Nx(), p.epsilon.Ny());
  s << fmtTitle % "Element-wise LJ Params" << endl;

  for(auto i : irange(0u, dmy.Nx())) for(int j : irange(0u, dmy.Ny())) dmy(i,j) = p.epsilon(i,j)/4.0/6.0;
  s << fmtSub % "Epsilon" << endl;  printArray(s, fmtHead, fmtDbl, dmy);   s << '\n';

  for(auto i : irange(0u, dmy.Nx())) for(int j : irange(0u, dmy.Ny())) dmy(i,j) = sqrt(p.sigma2(i,j));
  s << fmtSub % "Sigma"   << endl;  printArray(s, fmtHead, fmtDbl, dmy);     s << '\n';

  for(auto i : irange(0u, dmy.Nx())) for(int j : irange(0u, dmy.Ny())) dmy(i,j) = sqrt(p.rcut2(i,j));
  s << fmtSub % "Rcut"    << endl;  printArray(s, fmtHead, fmtDbl, dmy); s << endl;

  for(auto i : irange(0u, dmy.Nx())) for(int j : irange(0u, dmy.Ny())) dmy(i,j) = sqrt(p.rcut2(i,j)/p.sigma2(i,j));
  s << fmtSub % "Rcut/Sigma"    << endl;  printArray(s, fmtHead, fmtDbl, dmy); s << endl;

  for(auto i : irange(0u, dmy.Nx())) for(int j : irange(0u, dmy.Ny())) dmy(i,j) = p.fcut(i,j);
  s << fmtSub % "Fshift" << endl; printArray(s, fmtHead, fmtDbl, dmy); s << '\n';

  for(auto i : irange(0u, dmy.Ny())){
    for(int j : irange(0u, dmy.Ny())){
      p.elemForce(i,j,p.rcut2(i,j),dmy(i,j));
      dmy(i,j) *= p.sigma2(i,j);
    };
  }
  s << fmtSub % "F(Rcut)*Sigma^2" << endl; printArray(s, fmtHead, fmtDbl, dmy); s << '\n';

  s << fmtDataDbl % "Max(Rcut)" % sqrt(p.RCUT2); s << '\n';
  s << fmtDataStr % "Truncation Scheme" % (SWITCH[OPT::FORCE_SHIFTED] ? "Shifted Forces" : "Shifted Potential");
  return s;
  }*/


// Output Cell Type Parameters
std::ostream& operator << (std::ostream& s, const CellParams &p){
  using std::endl;
  s << fmtDataInt % "Number of Elements" % p.numElems << endl;
  s << fmtHead    % "Element Names";
  for(auto it = p.mapNames.begin(); it != p.mapNames.end(); it++) s << format("%s [%d]  ") % (it->first) % (it->second);
  s << "\n" << "#\n";
  s << fmtHead    % "Unique Element Types";     printArray(s, fmtInt, p.elemType); s << '\n';
  s << fmtDataInt % "Number of Bonds"    % p.numBonds << endl;
  s << fmtDataStr % "Contact Inhibited"  % (p.CIL ? "YES" : "NO") << endl;
  s << fmtDataStr % "Dividing" % (p.DIV ? "YES" : "NO") << endl;
  s << fmtDataStr % "Div. Swelling" % (p.divisionSwelling ? "YES" : "NO") << endl;
  s << fmtDataStr % "Apoptosis" % (p.APO ? "YES" : "NO") << endl;

  if(p.DIV){
    s << "#\n";
    s << fmtHead  % "Migration Tau"; s << fmtDataDbl0 % "mu" % p.migrationTau.mu; s << fmtDataDbl0 % "sigma" % p.migrationTau.sigma << endl;
    s << fmtHead  % "Division Tau" ; s << fmtDataDbl0 % "mu" % p.divisionTau.mu;  s << fmtDataDbl0 % "sigma" % p.divisionTau.sigma  << endl;
    s << fmtHead  % "Division R0";    s << fmtDbl % p.divisionR0 << endl;
    s << fmtHead  % "Division Rmin";  s << fmtDbl % sqrt(p.divisionR2min) << endl;
    s << "#\n";
  }
  if(p.APO){
    s << "#\n";
    s << fmtHead  % "Apoptosis"; s << fmtDataDbl0 % "Rate" % p.apopRate << endl;
    s << "#\n";
  }
  s << fmtDataStr % "External Force" % (p.EXTERNAL ? "YES" : "NO") << endl;
  if(p.EXTERNAL){
    s << "#\n";
    s << fmtHead % "Force"; s << fmtDataDbl0 % "x" % p.fExt[0]; s << fmtDataDbl0 % "y" % p.fExt[1] << endl;
    s << "#\n";
  }
  s << fmtHead    % "Motility";     printArray(s, fmtDbl, p.m);       s << '\n';
  s << fmtHead    % "Friction";     printArray(s, fmtDbl, p.zeta);    s << '\n';
  s << fmtHead    % "sqrt(Noise)";  printArray(s, fmtDbl, p.dNoise, sqrt(1.0/paramsMD.hdt));  s << '\n';
  s << "#\n";
  s << fmtHead    % "Sigma Core";   printArray(s, fmtDbl, p.sigmaCore);    s << '\n';
  s << fmtHead    % "Sigma Well";   printArray(s, fmtDbl, p.sigmaWell);    s << '\n';
  s << fmtHead    % "Epsilon Core"; printArray(s, fmtDbl, p.epsilonCore);  s << '\n';
  s << fmtHead    % "Epsilon Well"; printArray(s, fmtDbl, p.epsilonWell);  s << '\n';
  s << "#\n";
  if(p.numElems > 1){
    s << fmtHead  % "FENE Kappa" << endl; printArray(s, fmtHead, fmtDbl, p.kappa); s << '\n';
    s << fmtHead  % "FENE R2max" << endl; printArray(s, fmtHead, fmtDbl, p.r2max); s << '\n';
    s << fmtHead  % "FENE R2max (Division)" << endl; printArray(s, fmtHead, fmtDbl, p.divisionR2max); s << '\n';
    s << fmtHead  % "RSS" << endl; printArray(s, fmtHead, fmtDbl, p.rss); s << '\n';
  }
  s << "#";
  return s;
}

// Output Colony Parameters
std::ostream& operator << (std::ostream& s, const CellColony &c){
  using std::endl;

  {
    s << fmtTitle % "Cell Parameters" << endl;
    for(auto cellType : irange(0, c.specs)){
      auto cellName = c.nameCells[cellType];
      string dmy = "Species " + cellName;
      s << fmtSub % dmy << endl;
      s << fmtDataInt % "Species ID"   % cellType << endl;
      s << c.paramCells[cellType] << endl;
      s << fmtDataInt % "Total Cells" % c.totalCells[cellType];  s << fmtDataInt0 % "Begin UID" % c.cumulCells[cellType]; s << fmtDataInt0 % "End UID" % (c.cumulCells[cellType+1]); s << '\n';
      s << fmtDataInt % "Total Elems" % c.totalElems[cellType];  s << fmtDataInt0 % "Begin UID" % c.cumulElems[cellType]; s << fmtDataInt0 % "End UID" % (c.cumulElems[cellType+1]); s << '\n';
    }
  }
  s << '#';

  {
    std::ofstream ofs;
      std::string sPath = paramsOut.dirName+"/RunParams/";
      mode_t nMode = 0733; // UNIX style permissions
      mkdir(sPath.c_str(),nMode); // can be used on non-Windows
    ofs.open(paramsOut.dirName + "/RunParams/" + paramsOut.prjName + ".InitConf.dat", std::ofstream::out | std::ofstream::trunc);
    format fmtTop("# %8s %8s %8s %8s %8s %8s %12s %12s %8s %8s %8s"); // DIMENSION WARNING
    format fmtCell("  %8d %8d %8d %8d %8d %8d %12.5e %12.5e %8d %8d %8d");
    ofs << fmtTitle % "Initial Configuration"; ofs << '\n';
    ofs << fmtTop % "cellType" % "cellID" % "cellUID" % "elemType" % "elemID" % "elemUID" % "X" % "Y" % "State" % "t0" % "ts"<< endl;
    for(auto cellType : irange(0, c.specs)){ // over species
      auto   &pos = c.pos[cellType];
      auto   $vel = c.vel[cellType];
      auto   &status = c.cellState[cellType];
      auto   &t0     = c.cellT0[cellType];
      auto   &ts     = c.cellTSwitch[cellType];

      for(auto cellID : irange(0, c.totalCells[cellType])){ // over cells
	cellIDs cx(cellType, cellID);
	auto cellUID = c.cell2UID(cx);
	auto c0      = c.UID2Cell(cellUID);
	run_assert(c0.type == cx.type && c0.id == cx.id, "Error in UID2Cell");

	for(auto elemID : irange(0, c.paramCells[cellType].numElems)){ // over elements
	  auto elemType = c.paramCells[cellType].elemType[elemID];
	  elemIDs ex(cx.type, cx.id, elemType, elemID);

	  auto elemUID  = c.elem2UID(ex);
	  auto e0       = c.UID2Elem(elemUID);
	  run_assert(e0.cell.type == cellType && e0.cell.id == cellID && e0.type == elemType && e0.id == elemID, "Error in UID2Elem");
	  ofs << fmtCell % cellType % cellID % cellUID % elemType % elemID % elemUID % pos(cellID, elemID, 0) % pos(cellID, elemID, 1)
	    % status(cellID) % t0(cellID) % ts(cellID) << endl;
	}
      }
      ofs << "#\n";
    }
    ofs.close();
  }
  return s;
}
std::ostream& operator << (std::ostream& s, const OutputParams &o){
  using std::endl;
  s << fmtTitle % "Output Parameters" << endl;
  s << fmtDataStr % "Directory" % o.dirName << endl;
  s << fmtDataStr % "Name" % o.prjName;
  return s;
}
