#include "Container.hpp"
void LinkListBase::init(const MDutils::BoundingBox &box, const double &cellWidth){
    assert(DIM == box.dim());
    int    ns[MAX_DIM]    = {1, 1, 1};
    double lbox[MAX_DIM]  = {0.0, 0.0, 0.0};
    bool   pbc[MAX_DIM]   = {false, false, false};
    box.length(lbox);
    box.isPeriodic(pbc);

    // initialize link list
    if(box.dim() == 2){
      link = new MDutils::LinkListOrtho2D;
    }else if(box.dim() == 3){
      link = new MDutils::LinkListOrtho3D;
    }else{
      run_assert(false, "Only 2d/3d support for linklists");
    }
    link->get_opt_size(ns, cellWidth, lbox);
    numCells = link->init(numNeighbors, ns, lbox, pbc);

    // initialize neighbors id list
    neighborIds.Allocate(numNeighbors, DIM);
    link->get_cells(NULL, neighborIds());

    // initialize head list
    _head.resize(numCells);
    head.Dimension(numCells, _head.data());

    if(SWITCH[OPT::LINKLIST_SORTED]){
      _ihead.resize(numCells);
      ihead.Dimension(numCells, _ihead.data());
    }

    // initialize link list
    _list.resize(0);
    list.Dimension(0, _list.data());
}

void LinkListBase::reset(const MDutils::BoundingBox &box, const double &cellWidth){
  int ns[MAX_DIM]      = {1, 1, 1};
  double lbox[MAX_DIM] = {0.0, 0.0, 0.0};
  box.length(lbox);

  // reset link list
  link->get_opt_size(ns, cellWidth, lbox);
  numCells = link->reset(ns, lbox);

  // resize head list
  if(numCells > _head.size()){
    _head.resize(numCells);
    if(SWITCH[OPT::LINKLIST_SORTED]) _ihead.resize(numCells);
  }

  head.Dimension(numCells, _head.data());
  if(SWITCH[OPT::LINKLIST_SORTED]) ihead.Dimension(numCells, _ihead.data());
}

int LinkList::populate(const CellColony &col){

  // Make sure list can hold all elements
  auto totalElems = col.getTotalElems();
  if(totalElems > _list.size()) _list.resize(totalElems);
  list.Dimension(totalElems, _list.data());

  link->reset_list(head, list, totalElems); // set head/link  = -1
  for(auto cellSpec : irange(0, col.getCellTypes())){
    link->populate_list(col.pos[cellSpec], head, list,
		       col.getNumElems(cellSpec), col.cumulElems[cellSpec]);
  }
  int endCell = numCells;
  if(SWITCH[OPT::LINKLIST_SORTED]){
    std::iota(ihead.begin(), ihead.end(), 0);
    array1<int> headcp(head.Size(), head.begin()); // need local copy for lambda capture
    auto comp = [&headcp](const int &a, const int &b)->bool{return headcp[a] > headcp[b];};
    std::sort(ihead.begin(), ihead.end(), comp);
    for(endCell = 0; endCell < numCells && head[ihead[endCell]] > -1; endCell++);
  }
  return endCell;
}

int CellLinkList::populate(const CellColony &col){

  // Make sure list can hold all Cells
  auto totalCells = col.getTotalCells();
  if(totalCells > _list.size()) _list.resize(totalCells);
  list.Dimension(totalCells, _list.data());

  // Insert cells into the linked list by considering the position of their center of mass
  link->reset_list(head, list, totalCells); // set head/link  = -1
  for(auto cellSpec : irange(0, col.getCellTypes())){
    link->populate_list(col.com[cellSpec], head, list,
                       col.getNumCells(cellSpec), col.cumulCells[cellSpec]);
  }
  int endCell = numCells;
  if(SWITCH[OPT::LINKLIST_SORTED]){
    std::iota(ihead.begin(), ihead.end(), 0);
    array1<int> headcp(head.Size(), head.begin()); // need local copy for lambda capture
    auto comp = [&headcp](const int &a, const int &b)->bool{return headcp[a] > headcp[b];};
    std::sort(ihead.begin(), ihead.end(), comp);
    for(endCell = 0; endCell < numCells && head[ihead[endCell]] > -1; endCell++);
  }
  return endCell;
}
