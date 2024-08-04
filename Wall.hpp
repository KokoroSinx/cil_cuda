#ifndef WALL_HPP
#include "Format.hpp"

class Wall{
protected:
  static const double CUTOFF; // effectively a force cutoff: distance for which the force is calculated in case that the particle is closer than CUTOFF to the wall.
  double rcut;      // largest distance for which the wall-particle interaction is considered
  double epsilon;   // energy scale of the potential
  bool   positive;  // sense of the interaction: true keeps particles inside the enclosed area, false keeps them outside

public:
  Wall(const double _rcut, const double _epsilon, const bool _positive): rcut(_rcut), epsilon(_epsilon), positive(_positive){};
  virtual void addForce(const double &sigma, const double* r, double* f, double &p) = 0;
  //virtual void addHZForce(const double &sigma, const double* r, double* f, double &p, const double V, const double E) = 0;
  virtual void dump(std::ostream &s) = 0;
};
class CircularWall:public Wall{
private:
  double pos[MAX_DIM];
  double radius;
public:
  CircularWall(const double* _pos, const double _radius, const double _rcut, const double _epsilon, const bool _positive):
    Wall(_rcut, _epsilon, _positive), pos{0.0, 0.0, 0.0}, radius(_radius){
      for(auto d : irange(0, DIM)) pos[d] = _pos[d];
    }

  void addForce(const double &sigma, const double* r, double* f, double &p){
    double dr[MAX_DIM];
    double r0 = 0.0;            // distance from wall center
    for(auto d : irange(0, DIM)){
       dr[d] = r[d]-pos[d];
       r0   += dr[d]*dr[d];
    }
    double f0 = r0 = sqrt(r0);
    r0 = (positive ? (r0 - radius) : (radius - r0))/sigma;  // distance from wall / sigma
    if(r0 <= rcut){
      r0         = MAX(r0, CUTOFF);

      f0  = 1.0 / f0;
      f0  = (positive ? f0 : -f0);
      for(auto d : irange(0, DIM)) dr[d] *= f0;                // unit radial vector pointing inside or outside

      double ir2 = 1.0/(r0*r0);                                // (sigma/r)^2      [r = distance to wall]
      f0         = ir2*ir2*ir2;                                // (sigma/r)^n [n=6]
      f0         = epsilon*(2.0*f0 - 1.0)*f0*ir2/SQ(sigma);    // -24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r^2

      for(auto d : irange(0, DIM)) f[d] += f0*dr[d];
    }
  }
  void dump(std::ostream &s){
    using std::endl;
    const string rlabel[MAX_DIM] = {"x0", "y0", "z0"};
    s << fmtDataStr % "Type" % "Circular Wall" << endl;
    for(auto d : irange(0, DIM)) s << fmtDataDbl % rlabel[d] % pos[d];
    s << fmtDataDbl % "Radius" % radius << endl;
    s << fmtDataDbl % "rcut" % rcut << endl;
    s << fmtDataDbl % "epsilon" % epsilon << endl;
    s << fmtDataStr % "Direction" % (positive ? "positive" : "negative") << endl;
  }
};

class FlatWall:public Wall{
private:
  double x0; // position of all
  int    d;  // axis of perpendicular vector (x=0, y=1, z=2)
public:
  FlatWall(const double _x0, const int _d, const double _rcut, const double _epsilon, const bool _positive): Wall(_rcut, _epsilon, _positive), x0(_x0), d(_d){}
  void addForce(const double &sigma, const double* r, double* f, double &p){
    double dr = (positive ? (r[d] - x0) : (x0 - r[d]))/sigma;
    if(dr <= rcut){
      dr        = MAX(dr, CUTOFF);
      double ir2= 1.0 / (dr*dr);
      double f0 = ir2*ir2*ir2;
      f0        = epsilon*(2.0*f0 - 1.0)*f0*ir2/SQ(sigma);
      f0        = (positive ? f0 : -f0);
      p        += 2*ABS(f0)/(4*Constants::PI*SQ(sigma/2));
      f[d]     += f0;
    }
  }
  void addHZForce(const double &sigma, const double* r, double* f, double &p, const double V, const double E) {
      double dr = (positive ? (r[d]-x0-sigma) : (x0 - r[d]-sigma));
      if(dr < 0){
          auto h = ABS(dr);
          auto Aij = Constants::PI*h*sigma/2;
          double f0 = pow(h, 1.5)/((0.75) * ((1 - SQ(V)) / E + (1 - SQ(V)) / E) * (sqrt((1 / sigma) + (1 / sigma))));
          auto temp = Aij==0?0:f0/(4*Constants::PI*POW3(sigma/2)/3);
          f0        = (positive ? f0 : -f0);
          p        += 2*ABS(f0)/(4*Constants::PI*POW3(sigma/2)/3);
          f[d]     += f0;
      }
  }
  void dump(std::ostream &s){
    using std::endl;
    const string lbl[MAX_DIM] = {"X-Wall", "Y-Wall", "Z-Wall"};
    s << fmtDataStr % "Type" % lbl[d] << endl;
    s << fmtDataDbl % "r0" % x0 << endl;
    s << fmtDataDbl % "rcut" % rcut << endl;
    s << fmtDataDbl % "epsilon" % epsilon << endl;
    s << fmtDataStr % "Direction" % (positive ? "positive" : "negative")<< endl;
  }
};


class PartFlatWall:public Wall{
private:
    double x0; // position of all
    int    d;// axis of perpendicular vector (x=0, y=1, z=2)
    double e1;
    double e2;
public:
    PartFlatWall(const double _x0, const int _d, const double _e1, const double _e2, const double _rcut, const double _epsilon, const bool _positive): Wall(_rcut, _epsilon, _positive), x0(_x0), d(_d), e1(_e1), e2(_e2){}
    void addForce(const double &sigma, const double* r, double* f, double &p){
        double dr = (positive ? (r[d] - x0) : (x0 - r[d]))/sigma;
        int d0 = (d==1?0:1);
        if((dr <= rcut)){
            dr        = MAX(dr, CUTOFF);
            double ir2= 1.0 / (dr*dr);
            double f0 = ir2*ir2*ir2;
            f0        = epsilon*(2.0*f0 - 1.0)*f0*ir2/SQ(sigma);
            f0        = (positive ? f0 : -f0);
            f[d]     += f0;
        }
    }

  void addHZForce(const double &sigma, const double* r, double* f, double &p, const double V, const double E) {
      auto A = sigma;
  }
    void dump(std::ostream &s){
        using std::endl;
        const string lbl[MAX_DIM] = {"X-Wall", "Y-Wall", "Z-Wall"};
        s << fmtDataStr % "Type" % lbl[d] << endl;
        s << fmtDataDbl % "r0" % x0 << endl;
        s << fmtDataDbl % "rcut" % rcut << endl;
        s << fmtDataDbl % "epsilon" % epsilon << endl;
        s << fmtDataStr % "Direction" % (positive ? "positive" : "negative")<< endl;
    }
};

class HarmonicWall:public Wall{
private:
    double x0; // position of all
    int    d;  // axis of perpendicular vector (x=0, y=1, z=2)
public:
    HarmonicWall(const double _x0, const int _d, const double _rcut, const double _epsilon, const bool _positive): Wall(_rcut, _epsilon, _positive), x0(_x0), d(_d){}
    void addForce(const double &sigma, const double* r, double* f, double &p){

        double dr = r[d]-x0;
        if(ABS(dr)<= rcut){
            double f0 = epsilon*dr*0.5;
            f[d]     -= f0;
        }
    }
  void addHZForce(const double &sigma, const double* r, double* f, double &p, const double V, const double E) {
      auto A = sigma;
  }
    void dump(std::ostream &s){
        using std::endl;
        const string lbl[MAX_DIM] = {"X-Wall", "Y-Wall", "Z-Wall"};
        s << fmtDataStr % "Type" % lbl[d] << endl;
        s << fmtDataDbl % "r0" % x0 << endl;
        s << fmtDataDbl % "rcut" % rcut << endl;
        s << fmtDataDbl % "epsilon" % epsilon << endl;
        s << fmtDataStr % "Direction" % (positive ? "positive" : "negative")<< endl;
    }
};

class WallParams{
  int num;
public:
  vector<string> names;
  vector<Wall*>  walls;
  WallParams():num(0),names(),walls(){}
  void addWall(const string &_name, Wall* _wall){
    num++;
    names.resize(num);
    walls.resize(num);
    names[num-1] = _name;
    walls[num-1] = _wall;
  }
  int size()const{return num;}
};

#endif
