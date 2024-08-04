#ifndef COLONY_HPP
#define COLONY_HPP

#include "Cell.hpp"
#include <list>
using namespace std;
/*! \file Colony.hpp
 \brief The colony keeps track of the properties of all the cells of multiple cell species, such as parameters, positions, forces.
 */

typedef void (*intraCellForce)(const CellParams &params, const int &state, const array2<double> &elemPos, const array2<double> &elemFrc, const array2<double> &elemVel,const array1<double> &Wiener,const array1<double> &sig);
typedef void (*dividingCellForce)(const double &rr, const double &sigmaTheta, const double &sigmaCore, const array2<double> &motherPos, const array2<double> &daughterPos);
typedef void (*dividingCellReset)(const CellParams &params, const array2<double> &motherPos);
typedef bool (*dividingCellCriteria)(const CellParams &params, const array2<double> &elemPos, const array1<double> &sigmaCore);

class CellColony{ // Cell Data
private:
    int                          specs;          // number of species

    vector<int>                  totalCells;     // total number of cells of each species/colony
    vector<int>                  totalElems;     // total number of subelements of each species/colony = numCells[specID] * paramElems[specID].numElems

    vector<int>                  maxCells;       // maximum number of cells of each type that can fit into pos/frc vector

    vector<vector<double>>       _sigmaCore;     // pointer to memory for sigmas : size[i] = cellCapacity[i]
    vector<vector<double>>       _sigmaTheta;    // pointer to memory for cell cycle progress : size[i] = cellCapacity[i]
    vector<vector<double>>       _sigmaCoreT;    // pointer to memory for target sigma : size[i] = cellCapacity[i]
    vector<vector<double>>       _sigmaMax;      // pointer to memory for max sigma : size[i] = cellCapacity[i]
    vector<vector<double>>       _pos;           // pointer to memory for positions : size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _posCellsDiv;   // pointer to memory for positions of dividing mother cell: size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _posCellsApp;   // pointer to memory for positions of apoptosis mother cell: size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _pos0;          // pointer to memory for temporary positions, used for iteration: size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _frc;           // pointer to memory for forces    : size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _tor;           // pointer to memory for torques   : size[i] = cellCapacity[i]
    vector<vector<double>>       _orien;
    vector<vector<double>>       _frcCon;        // pointer to memory for conservative forces : size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _frcDis;        // pointer to memory for dissipative forces : size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _frcRan;        // pointer to memory for random forces: size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _pressure;      // pointer to memory for random forces: size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _vel;           // pointer to memory for velocities : size[i] = cellCapacity[i] * DIM
    vector<vector<double>>       _com;           // pointer to memory for centers of mass: size[i] = numCells[i] * DIM


    vector<vector<double>>       _neighbors;     // pointer to memory for linklist neighbors

    vector<vector<double>>       _activeWiener;  // pointer to memory for active winer processes   : size[i] = numCells[i] * 2
    vector<vector<double>>       _passiveWiener; // pointer to memory for passive wiener processes : size[i] = numCells[i] * DIM

    vector<vector<int>>          _cellState;     // pointer to memory for cell states   : size[i] = numCells[i]
    vector<vector<int>>          _cellT0;        // pointer to memory for computation start time for cells  : size[i] = numCells[i]
    vector<vector<double>>       _cellTheta;
    vector<vector<int>>          _cellT0Div;
    vector<vector<int>>          _cellT0App;
    vector<vector<int>>          _cellTSwitch;
    vector<vector<int>>          _cellTPressure;
    vector<vector<double>>       _cellAngVel;
    // vector<vector<double>>       _cellAngel;
    vector<vector<double>>       _cellTRho;
    vector<vector<double>>       _cellTTheta;
    vector<vector<int>>          _divisionPair;   //

    // Output Buffers that can hold all elements in the system
    vector<float>                vectorBuffer;    // DIM * bufferSize
    vector<int>                  scalarBuffer;    // bufferSize
    vector<unsigned int>         scalarUnsignedBuffer;

public:
    vector<string>               nameCells;      ///< cell names
    vector<CellParams>           paramCells;     ///< parameters for each cell species
    vector<ElemParams>           paramElems;     ///< parameters for each element

    vector<int>                  cumulCells;     ///< cumulative number of cells for each species/colony    = cell UIDs
    vector<int>                  cumulElems;     ///< cumulative number of elements for each species/colony = element UIDs

    vector<intraCellForce>       migrationFunc;  ///< function pointers to intra-cellular motility forces

    vector<dividingCellForce>    divisionFunc;   ///< function pointers to division functions
    vector<dividingCellReset>    divisionFail;
    vector<dividingCellCriteria> divisionCheck;  ///< function pointers to check if cell should divide

    vector<array2<double>>       sigmaCore;      ///< sig(spec)    = <frc(i, j)> i,j in nCells(spec), nElems(spec)
    vector<array2<double>>       sigmaTheta;
    vector<array2<double>>       sigmaCoreT;    // target sigma
    vector<array2<double>>       sigmaMin;
    vector<array2<double>>       sigmaMax;
    vector<array2<double>>       intraDis;      ///< sig(spec)    = <frc(i, j)> i,j in nCells(spec), nElems(spec)

    vector<array3<double>>       pos;            ///< pos(spec)    = <pos(i, j, d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array3<double>>       pos0;
    vector<array3<double>>       frc;            ///< frc(spec)    = <frc(i, j, d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array1<double>>       tor;            ///< tor(spec)    = <tor(i)> i,j,d in nCells(spec), nElems(spec) torques
    vector<array1<double>>       orien;          ///< tor(spec)    = <tor(i)> i,j,d in nCells(spec), nElems(spec) orientation
    vector<array3<double>>       frcCon;         ///< frcCon(spec) = <frc(i, j, d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array3<double>>       frcDis;         ///< frcDis(spec) = <frc(i, j, d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array3<double>>       frcRan;         ///< frcRan(spec) = <frc(i, j, d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array1<double>>       pressure;       ///< frcRan(spec) = <frc(i, j, d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array3<double>>       vel;            ///< vel(spec)    = <vel(i, j ,d)> i,j,d in nCells(spec), nElems(spec), DIM
    vector<array2<double>>       com;            ///< centers of mass, com(spec)    = <com(i, j)> i,j in nCells(spec), DIM

    vector<array2<double>>       neighbors;      ///nCells, unspecific length of neighbor list

    vector<array2<double>>       activeWiener;   ///< W(spec) = <W(i,j)> i,j in nCells(spec), [radial, angular]
    vector<array2<double>>       passiveWiener;  ///< W(spec) = <W(i,j)> i,j in nCells(spec), [x, y]

    vector<array1<int>>          cellState;      ///< crawling/dividing state of cells
    vector<array1<int>>          cellT0;         ///< start time for cells
    vector<array1<int>>          cellTSwitch;    ///< transition time for cells
    vector<array1<int>>          cellTPressure;  ///< time for cells under high pressure
    vector<array1<double>>       cellAngVel;     ///< angular velocity in a cell cycle
    // vector<array1<double>>       cellAngel;      ///< current angel in a cell cycle
    vector<array1<double>>       cellTRho;     	 ///< angular velocity in a cell cycle
    vector<array1<double>>       cellTTheta;     ///< current angel in a cell cycle
    vector<array1<int>>          divisionPair;   ///< if it's a dividing cell, then what's the pair cell

    vector<array3<double>>       posCellsDiv;    ///< positions of cells divided.
    vector<array3<double>>       posCellsApp;    ///< positions of cells divided.
    vector<int>                  nCellsDiv;      ///< instant number of cells divided.
    vector<int>                  nCellsApp;      ///< instant number of cells apoptosised.
    vector<int>                  cumulCellsDiv;  ///< cumulative number of cells divided.
    vector<int>                  cumulCellsApp;  ///< cumulative number of cells apoptosised.
    vector<array1<int>>          cellT0Div;      ///< instant computer time of cells divided.
    vector<array1<int>>          cellT0App;      ///< instant computer time of cells apoptosised.

    friend std::ostream& operator<<(std::ostream&, const CellColony&);

    void init(vector<string> _names);

    // name of the cell type/species -> cell type/spec ID for one species
    int name2CellType(const string &name) const;

    // local cell id ->  global unique cell id (for all species)
    int cell2UID(const cellIDs &c) const;

    // local element id -> global unique element id (for all cell species and all elements)
    int elem2UID(const elemIDs &e) const;

    // global cell id -> local cell id for specific type
    cellIDs UID2Cell(const int &uid) const;

    // global element id -> local element id for specific cell type
    elemIDs UID2Elem(const int &uid) const;


    void synchronize();

    void reserveNumCells(const int specID, const int reserveNum);
    void setNumCells(const int specID, const int num);
    void setNumCellsDiv(const int specID, const int numDiv, const int ts);
    void setNumCellsApp(const int specID, const int numApp, const int ts);
    void resetNumCellsDiv(const int specID);
    void resetNumCellsApp(const int specID);
    int addCell(const int specID);
    int removeCell(const int specID, const int cellID);

    inline double getMaxSigma()const{
        double sig = 0.0;
        for(auto cellType : irange(0, specs)){
            auto &sigmas = paramCells[cellType].sigmaCore;
            sig = MAX(sig, *std::max_element(sigmas.begin(), sigmas.end()));
        }
        return sig;
    }
    inline double getMinSigma()const{
        double sig =  Constants::MAX_MP;
        for(auto cellType : irange(0, specs)){
            auto &sigmas = paramCells[cellType].sigmaCore;
            sig = MIN(sig, *std::min_element(sigmas.begin(), sigmas.end()));
        }
        return sig;
    }

    inline int getCellTypes()const {return specs;}

    inline int getElemTypes()const {
        int count = 0;
        for(auto &dmy : paramCells) count += dmy.numElems;
        return count;
    }

    /*!
     Total number of cells in the simulation for the given species ID.
     */
    inline int getNumCells(const int specID) const{assert(specID >= 0 && specID < specs); return totalCells[specID];}
    inline int getNumCellsDiv(const int specID) const{assert(specID >= 0 && specID < specs); return nCellsDiv[specID];}
    inline int getNumCellsApp(const int specID) const{assert(specID >= 0 && specID < specs); return nCellsApp[specID];}
    inline int getCumCellsDiv(const int specID) const{assert(specID >= 0 && specID < specs); return cumulCellsDiv[specID];}
    inline int getCumCellsApp(const int specID) const{assert(specID >= 0 && specID < specs); return cumulCellsApp[specID];}
    /*!
    Total number of elements in the simulation for the given species ID.
    */
    inline int getNumElems(const int specID) const{assert(specID >= 0 && specID < specs); return totalElems[specID];}

    /*!
     Total number of cells in the simulation.
     */
    inline int getTotalCells()const{return std::accumulate(totalCells.begin(), totalCells.end(), 0);}

    /*!
     Total number of elements in the simulation.
     */
    inline int getTotalElems()const{return std::accumulate(totalElems.begin(), totalElems.end(), 0);}

    /// Calculate the centers of mass
    void calculComs();

    inline float        * getVectorBuffer(){return vectorBuffer.data();}
    inline int          * getScalarBuffer(){return scalarBuffer.data();}
    inline unsigned int * getScalarUnsignedBuffer(){return scalarUnsignedBuffer.data();}
};

#endif
