#include "Colony.hpp"
#include "Container.hpp"

extern Container container;

/*! \file Colony.cpp
 \brief The colony keeps track of the properties of all the cells of multiple cell species, such as parameters, positions, forces.
 */

//
int CellColony::name2CellType(const string &name) const{
    auto startNames = nameCells.begin();
    auto endNames = nameCells.end();
    int numCellTypes = getCellTypes();
    auto spec_i = std::distance(startNames, std::find(startNames, endNames, name));
    run_assert(spec_i >= 0 && spec_i < numCellTypes, "Unknown cell type with name " + name);
    run_assert(nameCells[spec_i] == name);
    return spec_i;
}

// local cell id ->  global unique cell id (for all species)
int CellColony::cell2UID(const cellIDs &c) const{
    assert(c.type >= 0 && c.type < specs);
    assert(c.id >= 0 && c.id < totalCells[c.type]);
    assert(cumulCells[c.type] + c.id < cumulCells[c.type + 1]);
    return cumulCells[c.type] + c.id;
}
// local element id -> global unique element id (for all cell species and all elements)
int CellColony::elem2UID(const elemIDs &e) const{
    assert(e.cell.type >= 0 && e.cell.type < specs);
    assert(e.cell.id >= 0 && e.cell.id < totalCells[e.cell.type]);
    assert(e.id >= 0 && e.id < totalElems[e.cell.type]);
    assert(cumulElems[e.cell.type] + paramCells[e.cell.type].numElems*e.cell.id + e.id < cumulElems[e.cell.type+1]);
    return cumulElems[e.cell.type] + paramCells[e.cell.type].numElems*e.cell.id + e.id;
}

// global cell id -> local cell id for specific type
cellIDs CellColony::UID2Cell(const int &uid) const{
    cellIDs cell;
    auto it = std::lower_bound(cumulCells.begin(), cumulCells.end(), uid);
    if((*it) != uid) it--;
    cell.type  = (it - cumulCells.begin());
    cell.id  = uid - *(it);

    assert(cell.type >= 0 && cell.type < specs);
    assert(cell.id >= 0 && cell.id < totalCells[cell.type]);
    assert(uid == cell2UID(cell));
    return cell;
}
// global element id -> local element id for specific cell type
elemIDs CellColony::UID2Elem(const int &uid)const{
    elemIDs elem;
    auto it = std::lower_bound(cumulElems.begin(), cumulElems.end(), uid);
    if((*it) != uid) it--;
    elem.cell.type = (it - cumulElems.begin());

    auto &nelem = paramCells[elem.cell.type].numElems;
    elem.cell.id   = (uid - *(it)) / nelem;
    elem.id   = uid - (elem.cell.id*nelem + (*it));
    elem.type = paramCells[elem.cell.type].elemType[elem.id];

    assert(elem.cell.type >= 0 && elem.cell.type < specs);
    assert(elem.cell.id >= 0 && elem.cell.id < totalCells[elem.cell.type]);
    assert(elem.id >= 0 && elem.id < totalElems[elem.cell.type]);
    assert(uid == elem2UID(elem));
    return elem;
}

void CellColony::init(vector<string> _names){
    {
        std::sort(_names.begin(), _names.end());
        specs = _names.size();
        std::unique(_names.begin(), _names.end());
        run_assert(specs > 0, "Must have at least one species!");
        run_assert(specs == _names.size(), "Species must have unique names");
        nameCells = std::move(_names);
    }

    //Private Data
    {
        totalCells.resize(specs, 0);
        totalElems.resize(specs, 0);

        maxCells.resize(specs, 0);

        _sigmaCore.resize(specs);
        _sigmaTheta.resize(specs);
        _sigmaCoreT.resize(specs);
        _pos.resize(specs);
        _pos0.resize(specs);
        _posCellsDiv.resize(specs);
        _posCellsApp.resize(specs);
        _tor.resize(specs);
        _orien.resize(specs);
        _frc.resize(specs);
        _frcDis.resize(specs);
        _frcRan.resize(specs);
        _pressure.resize(specs);
        _vel.resize(specs);
        _com.resize(specs);

        _neighbors.resize(specs);

        _activeWiener.resize(specs);
        _passiveWiener.resize(specs);

        _cellState.resize(specs);
        _cellT0.resize(specs);
        _cellT0Div.resize(specs);
        _cellT0App.resize(specs);
        _cellTSwitch.resize(specs);
        _cellTPressure.resize(specs);
        _cellAngVel.resize(specs);
        // _cellAngel.resize(specs);
        _cellTRho.resize(specs);
        _cellTTheta.resize(specs);
        _divisionPair.resize(specs);
    }

    //Public Data
    {
        paramElems.resize(0);
        paramCells.resize(specs);

        cumulCells.resize(specs+1, 0);
        cumulElems.resize(specs+1, 0);

        migrationFunc.resize(specs, NULL);
        divisionFunc.resize(specs, NULL);
        divisionFail.resize(specs, NULL);
        divisionCheck.resize(specs, NULL);

        sigmaCore.resize(specs);
        sigmaTheta.resize(specs);
        sigmaCoreT.resize(specs);
        pos.resize(specs);
        pos0.resize(specs);
        posCellsDiv.resize(specs);
        posCellsApp.resize(specs);
        tor.resize(specs);
        orien.resize(specs);
        frc.resize(specs);
        frcDis.resize(specs);
        frcRan.resize(specs);
        pressure.resize(specs);
        vel.resize(specs);
        com.resize(specs);

        neighbors.resize(specs);

        activeWiener.resize(specs);
        passiveWiener.resize(specs);

        cellState.resize(specs);
        cellT0.resize(specs);
        cellT0Div.resize(specs);
        cellT0App.resize(specs);
        cellTSwitch.resize(specs);
        cellTPressure.resize(specs);
        cellAngVel.resize(specs);
        // cellAngel.resize(specs);
        cellTRho.resize(specs);
        cellTTheta.resize(specs);
        divisionPair.resize(specs);
        cumulCellsDiv.resize(specs+1,0);
        cumulCellsApp.resize(specs+1,0);
        nCellsDiv.resize(specs, 0);
        nCellsApp.resize(specs, 0);
    }
}

void CellColony::synchronize(){
    // Synchronize Cumulative sums
    cumulCells[0] = cumulElems[0] = 0;
    for(auto i : irange(0, specs)){
        cumulCells[i+1] = cumulCells[i] + totalCells[i];
        cumulElems[i+1] = cumulElems[i] + totalElems[i];
    }

    // Synchronize Particle Buffers
    unsigned int newSize = cumulElems[specs];
    if(newSize > scalarBuffer.size()){
        newSize = static_cast<unsigned int>(newSize*1.5);
        vectorBuffer.resize(3*newSize, 0.0);
        scalarBuffer.resize(newSize, 0);
        scalarUnsignedBuffer.resize(newSize, 0);
    }
}

/*!
  Make sure we have enough space allocated to fit reserveNum of cells of type specID
 */
void CellColony::reserveNumCells(const int specID, const int reserveNum){
    assert(specID >= 0 && specID < specs);
    if((reserveNum > maxCells[specID])||(nCellsDiv[specID]>maxCells[specID])||(nCellsApp[specID]>maxCells[specID])){ // Increase array sizes if necessary
        auto nCapacity = maxCells[specID] = static_cast<int>(reserveNum*1.5);  // max cells
        auto nCells    = totalCells[specID];      // current number of cells
        auto numCellsDiv = nCellsDiv[specID];
        auto numCellsApp = nCellsApp[specID];
        auto nElems    = paramCells[specID].numElems; // elems per cell
        auto nSize     = nCapacity*nElems;        // max size for the 2d arrays
        auto nSizeVec  = nCapacity*nElems*DIM;    // max size for the 3d arrays

        _sigmaCore[specID].resize(nSize, 0.0);
        _sigmaTheta[specID].resize(nSize, 0.0);
        _sigmaCoreT[specID].resize(nSize, 0.0);
        _pos[specID].resize(nSizeVec, 0.0);
        _pos0[specID].resize(nSizeVec, 0.0);
        _tor[specID].resize(nCapacity, 0.0);
        _orien[specID].resize(nCapacity, 0.0);
        _frc[specID].resize(nSizeVec, 0.0);
        _frcDis[specID].resize(nSizeVec, 0.0);
        _frcRan[specID].resize(nSizeVec, 0.0);
        _pressure[specID].resize(nCapacity, 0.0);
        _vel[specID].resize(nSizeVec, 0.0);
        _com[specID].resize(nCapacity*DIM, 0.0);
        _neighbors[specID].resize(nCapacity, -1);
        _posCellsDiv[specID].resize(nSizeVec, 0.0);
        _posCellsApp[specID].resize(nSizeVec, 0.0);

        sigmaCore[specID].Dimension(nCells,  nElems, _sigmaCore[specID].data());
        sigmaTheta[specID].Dimension(nCells, nElems, _sigmaTheta[specID].data());
        sigmaCoreT[specID].Dimension(nCells,  nElems, _sigmaCoreT[specID].data());
        pos[specID].Dimension(nCells,  nElems, DIM, _pos[specID].data());
        pos0[specID].Dimension(nCells, nElems, DIM, _pos0[specID].data());
        posCellsDiv[specID].Dimension(numCellsDiv, nElems, DIM, _posCellsDiv[specID].data());
        posCellsApp[specID].Dimension(numCellsApp, nElems, DIM, _posCellsDiv[specID].data());

        tor[specID].Dimension(nCells, _tor[specID].data());
        orien[specID].Dimension(nCells, _orien[specID].data());
        frc[specID].Dimension(nCells,  nElems, DIM, _frc[specID].data());
        frcDis[specID].Dimension(nCells,  nElems, DIM, _frcDis[specID].data());
        frcRan[specID].Dimension(nCells,  nElems, DIM, _frcRan[specID].data());
        pressure[specID].Dimension(nCells, _pressure[specID].data());
        vel[specID].Dimension(nCells,  nElems, DIM, _vel[specID].data());
        com[specID].Dimension(nCells, DIM, _com[specID].data());
        neighbors[specID].Dimension(nCells, _neighbors[specID].data());

        _passiveWiener[specID].resize(nCapacity*DIM, 0.0);
        passiveWiener[specID].Dimension(nCells, DIM, _passiveWiener[specID].data());

        //if(nElems > 1){
        _activeWiener[specID].resize(nCapacity*ACTIVE_NOISE_DIM, 0.0);
        activeWiener[specID].Dimension(nCells,ACTIVE_NOISE_DIM, _activeWiener[specID].data());
        //}


        _cellState[specID].resize(nCapacity, CellState::Undefined);
        _cellT0[specID].resize(nCapacity, -1);
        _cellT0Div[specID].resize(nCapacity, -1);
        _cellT0App[specID].resize(nCapacity, -1);
        _cellTSwitch[specID].resize(nCapacity, -1);
        _cellTPressure[specID].resize(nCapacity, -1);
        _cellAngVel[specID].resize(nCapacity, -1);
        // _cellAngel[specID].resize(nCapacity, -1);
        _cellTTheta[specID].resize(nCapacity, -1);
        _divisionPair[specID].resize(nCapacity, -1);
        _cellTRho[specID].resize(nCapacity, -1);


        cellState[specID].Dimension(nCells,   _cellState[specID].data());
        cellT0[specID].Dimension(nCells,      _cellT0[specID].data());
        cellT0Div[specID].Dimension(nCells,      _cellT0Div[specID].data());
        cellT0App[specID].Dimension(nCells,      _cellT0App[specID].data());
        cellTSwitch[specID].Dimension(nCells, _cellTSwitch[specID].data());
        cellTPressure[specID].Dimension(nCells, _cellTPressure[specID].data());
        cellAngVel[specID].Dimension(nCells, _cellAngVel[specID].data());
        // cellAngel[specID].Dimension(nCells,  _cellAngel[specID].data());
        cellTTheta[specID].Dimension(nCells, _cellTTheta[specID].data());
        divisionPair[specID].Dimension(nCells, _divisionPair[specID].data());
        cellTRho[specID].Dimension(nCells, _cellTRho[specID].data());
    }
}

/*!
  Set number of cells of type specID to num
  Warning: Synchronize cumulative counters after sizes are modified!
 */
void CellColony::setNumCells(const int specID, const int num){
    assert(specID >= 0 && specID < specs);
    reserveNumCells(specID, num);
    auto nElems = paramCells[specID].numElems;
    totalCells[specID] = num;
    totalElems[specID] = num*nElems;
    sigmaCore[specID].Dimension(num,  nElems);
    sigmaTheta[specID].Dimension(num, nElems);
    sigmaCoreT[specID].Dimension(num,  nElems);
    pos[specID].Dimension(num,  nElems, DIM);
    pos0[specID].Dimension(num, nElems, DIM);
    frc[specID].Dimension(num,  nElems, DIM);
    tor[specID].Dimension(num);
    orien[specID].Dimension(num);
    frcDis[specID].Dimension(num,  nElems, DIM);
    frcRan[specID].Dimension(num,  nElems, DIM);
    pressure[specID].Dimension(num);
    vel[specID].Dimension(num,  nElems, DIM);
    com[specID].Dimension(num, DIM);
    neighbors[specID].Dimension(num);
    passiveWiener[specID].Dimension(num, DIM);
    //if(nElems > 1){
    activeWiener[specID].Dimension(num,  ACTIVE_NOISE_DIM);
    //}

    cellState[specID].Dimension(num);
    cellT0[specID].Dimension(num);
    cellTSwitch[specID].Dimension(num);
    cellTPressure[specID].Dimension(num);
    cellAngVel[specID].Dimension(num);
    // cellAngel[specID].Dimension(num);
    cellTTheta[specID].Dimension(num);
    divisionPair[specID].Dimension(num);
    cellTRho[specID].Dimension(num);
}
/*!
  Set number of divided/dead cells of type specID to numDiv/numApp
  Note: numDiv/numApp is the number of cells divided/died since last frame.
        cumulCellsDiv/App is the number of cells divided/died since beginning of the simulation.
 */
void CellColony::setNumCellsDiv(const int specID, const int numDiv, const int ts){
    auto nElems = paramCells[specID].numElems;
    posCellsDiv[specID].Dimension(numDiv, nElems, DIM);
    nCellsDiv[specID] = numDiv;
    cellT0Div[specID].Dimension(numDiv);
    cellT0Div[specID][numDiv-1]=ts;
    cumulCellsDiv[specID]+=1;
}
void CellColony::setNumCellsApp(const int specID, const int numApp, const int ts){
    auto nElems = paramCells[specID].numElems;
    posCellsApp[specID].Dimension(numApp, nElems, DIM);
    nCellsApp[specID] = numApp;
    cellT0App[specID].Dimension(numApp);
    cellT0App[specID][numApp-1]=ts;
    cumulCellsApp[specID]+=1;
}
/*!
  reset number of divided/dead cells of type specID to 0, after the info has been written to h5 file at each frame.
  Note: cumulCellsDiv/App will keep track of the index of accumulated divided/dead cells.
 */
void CellColony::resetNumCellsDiv(const int specID){
    auto nElems = paramCells[specID].numElems;
    posCellsDiv[specID].Dimension(0, nElems, DIM);
    nCellsDiv[specID] = 0;
    cellT0Div[specID].Dimension(0);
}
void CellColony::resetNumCellsApp(const int specID){
    auto nElems = paramCells[specID].numElems;
    posCellsApp[specID].Dimension(0, nElems, DIM);
    nCellsApp[specID] = 0;
    cellT0App[specID].Dimension(0);
}
/*!
  Convenience function to increase the number of cells of type specID by 1.
  Returns position of new element, always added at the end
 */
int CellColony::addCell(const int specID){
    assert(specID >= 0 && specID < specs);
    setNumCells(specID, totalCells[specID]+1);
    return totalCells[specID]-1;
}

/*!
 Deletes the cell of type specID and with id cellID from the colony
 Returns the position of the last cell, at the end of the list
 */
int CellColony::removeCell(const int specID, const int cellID){
    // Make sure that specID and cellID are valid IDs
    assert(specID >= 0 && specID < specs);
    assert(cellID >= 0 && cellID < getNumCells(specID));

    // Switch the to-be-deleted cell with the last cell in all the relevant arrays
    const int lastCell = getNumCells(specID) - 1;
    auto nElems = paramCells[specID].numElems;
    sigmaCore[specID][cellID] = sigmaCore[specID][lastCell];
    sigmaTheta[specID][cellID]= sigmaTheta[specID][lastCell];
    sigmaCoreT[specID][cellID] = sigmaCoreT[specID][lastCell];
    pos[specID][cellID]  = pos[specID][lastCell];
    pos0[specID][cellID] = pos0[specID][lastCell];
    tor[specID][cellID]  = tor[specID][lastCell];
    orien[specID][cellID]  = orien[specID][lastCell];
    frc[specID][cellID]  = frc[specID][lastCell];
    com[specID][cellID]  = com[specID][lastCell];
    passiveWiener[specID][cellID]  = passiveWiener[specID][lastCell];
    if(nElems > 1){
        activeWiener[specID][cellID]  = activeWiener[specID][lastCell];
    }
    cellState[specID][cellID]   = cellState[specID][lastCell];
    cellT0[specID][cellID]      = cellT0[specID][lastCell];
    cellTSwitch[specID][cellID] = cellTSwitch[specID][lastCell];
    cellTPressure[specID][cellID] = cellTPressure[specID][lastCell];
    cellAngVel[specID][cellID]  = cellAngVel[specID][lastCell];
    // cellAngel[specID][cellID]   = cellAngel[specID][lastCell];
    cellTTheta[specID][cellID]  = cellTTheta[specID][lastCell];
    divisionPair[specID][cellID]=divisionPair[specID][lastCell];
    cellTRho[specID][cellID]    = cellTRho[specID][lastCell];

    // decrease the number of cells by 1.
    setNumCells(specID, totalCells[specID]-1);
    return totalCells[specID]-1;
}

/*!
  Calculate the centers of mass of the cells, while accounting for periodic boundary conditions

  Account for periodic boundary conditions: Calculate the positions of the elements
        relative to the first one, apply the minimum image convention, and average those
        relative positions. Then, add this average to the first element and apply the periodic
         boundary convention to the result.
*/
void CellColony::calculComs() {
    for(auto ispec : irange(0, getCellTypes())){// over species
        const auto &spec_pos = pos[ispec];
        const auto &spec_com = com[ispec];
        const auto nCells = getNumCells(ispec);
        const auto nElems = paramCells[ispec].numElems;

#pragma  omp parallel for
        for(auto cellId : irange(0, nCells)){ // over cells of this species
            const auto cell_pos = spec_pos[cellId];
            const auto r0       = cell_pos[0];
            auto cell_com = spec_com[cellId];
            double rij[DIM];

            cell_com = 0.0;
            for(auto elemId : irange(0, nElems)){
                container.box.distance(rij, r0, cell_pos[elemId]); // rij = ri - r0
                for( auto d : irange(0,DIM)) cell_com[d] += rij[d];
            }
            cell_com /= nElems;
            container.box.updatePos(cell_com, r0);
        }
    }
}
