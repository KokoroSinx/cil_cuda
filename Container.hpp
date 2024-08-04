#ifndef CONTAINER_HPP
#define CONTAINER_HPP

#include "Common.hpp"
#include "Colony.hpp"

class Container;

/*!
  Base Link List Class
 */
class LinkListBase{
protected:
  vector<int>             _ihead;
  vector<int>             _head;
  vector<int>             _list;
public:
  MDutils::LinkListOrtho   *link;
  int                      numCells;
  int                      numNeighbors;
  array2<int>              neighborIds;

  array1<int>              ihead;
  array1<int>              head;
  array1<int>              list;
  void init(const MDutils::BoundingBox &box, const double &cellWidth);
  void reset(const MDutils::BoundingBox &box, const double &cellWidth);

  virtual int  populate(const CellColony &col) = 0; // populate list with all elements in colony
};

/*!
 \brief Link List for elements
 */
class LinkList : public LinkListBase{
public:
  int  populate(const CellColony &col);
};

/*!
\brief Link List for cells
*/
class CellLinkList : public LinkListBase{
public:
  int  populate(const CellColony &col);
};

/*!
  Container
 */
class Container{
public:
  MDutils::BoundingBox   box;
  CellLinkList           linkList;
};

#endif
