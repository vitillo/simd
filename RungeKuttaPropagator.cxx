/////////////////////////////////////////////////////////////////////////////////
// RungeKuttaPropagator.cxx, (c) ATLAS Detector software
/////////////////////////////////////////////////////////////////////////////////

#include <immintrin.h>

#define CROSS_SHUFFLE_201(Y1) _mm256_shuffle_pd(_mm256_permute2f128_pd(Y1, Y1, 0x01), Y1, 0xC);
#define CROSS_SHUFFLE_120(Y1) _mm256_permute_pd(_mm256_shuffle_pd(Y1, _mm256_permute2f128_pd(Y1, Y1, 0x01), 0x5), 0x6);

#include "TrkExUtils/RungeKuttaUtils.h"
#include "TrkExRungeKuttaPropagator/RungeKuttaPropagator.h"
#include "TrkSurfaces/ConeSurface.h"
#include "TrkSurfaces/DiscSurface.h"
#include "TrkSurfaces/PlaneSurface.h"
#include "TrkSurfaces/PerigeeSurface.h"
#include "TrkSurfaces/CylinderSurface.h"
#include "TrkSurfaces/StraightLineSurface.h"
#include "TrkGeometry/MagneticFieldProperties.h"
#include "TrkMagFieldInterfaces/IMagneticFieldTool.h"
#include "TrkEventPrimitives/ErrorMatrix.h"
#include "TrkEventPrimitives/CovarianceMatrix.h"
#include "TrkEventPrimitives/Transform3D.h"

#include "TrkParameters/AtaDisc.h"
#include "TrkParameters/Perigee.h"
#include "TrkParameters/AtaPlane.h"
#include "TrkParameters/AtaCylinder.h"
#include "TrkParameters/AtaStraightLine.h"
#include "TrkParameters/CurvilinearParameters.h"

#include "TrkParameters/MeasuredAtaDisc.h"
#include "TrkParameters/MeasuredPerigee.h"
#include "TrkParameters/MeasuredAtaPlane.h"
#include "TrkParameters/MeasuredAtaCylinder.h"
#include "TrkParameters/MeasuredAtaStraightLine.h"
#include "TrkParameters/MeasuredTrackParameters.h"
#include "TrkParameters/MeasuredCurvilinearParameters.h"

#include "TrkNeutralParameters/NeutralAtaPlane.h"
#include "TrkNeutralParameters/NeutralAtaDisc.h"
#include "TrkNeutralParameters/NeutralAtaCylinder.h"
#include "TrkNeutralParameters/NeutralAtaStraightLine.h"
#include "TrkNeutralParameters/NeutralPerigee.h"
#include "TrkNeutralParameters/NeutralCurvilinearParameters.h"

#include "TrkNeutralParameters/MeasuredNeutralAtaPlane.h"
#include "TrkNeutralParameters/MeasuredNeutralAtaDisc.h"
#include "TrkNeutralParameters/MeasuredNeutralAtaCylinder.h"
#include "TrkNeutralParameters/MeasuredNeutralAtaStraightLine.h"
#include "TrkNeutralParameters/MeasuredNeutralPerigee.h"
#include "TrkNeutralParameters/MeasuredNeutralCurvilinearParameters.h"

#include "TrkExUtils/IntersectionSolution.h"
#include "TrkExUtils/TransportJacobian.h"
#include "TrkPatternParameters/PatternTrackParameters.h"
#include "CLHEP/Geometry/Point3D.h"

/////////////////////////////////////////////////////////////////////////////////
// Constructor
/////////////////////////////////////////////////////////////////////////////////

Trk::RungeKuttaPropagator::RungeKuttaPropagator
(const std::string& p,const std::string& n,const IInterface* t) :  AthAlgTool(p,n,t)
{
  m_dlt               = .000200;
  m_helixStep         = 1.     ; 
  m_straightStep      = .01    ;
  m_usegradient       = false  ;
  m_magneticFieldTool = 0      ;
 
  declareInterface<Trk::IPropagator>(this);   
  declareInterface<Trk::IPatternParametersPropagator>(this);
  declareProperty("AccuracyParameter"  ,m_dlt          );
  declareProperty("MaxHelixStep"       ,m_helixStep    );
  declareProperty("MaxStraightLineStep", m_straightStep);
  declareProperty("IncludeBgradients"  , m_usegradient );
}

/////////////////////////////////////////////////////////////////////////////////
// initialize
/////////////////////////////////////////////////////////////////////////////////

StatusCode Trk::RungeKuttaPropagator::initialize()
{
  msg(MSG::INFO) << name() <<" initialize() successful" << endreq;
  return StatusCode::SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////
// finalize
/////////////////////////////////////////////////////////////////////////////////

StatusCode  Trk::RungeKuttaPropagator::finalize()
{
  msg(MSG::INFO) << name() <<" finalize() successful" << endreq;
  return StatusCode::SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////
// Destructor
/////////////////////////////////////////////////////////////////////////////////

Trk::RungeKuttaPropagator::~RungeKuttaPropagator(){}


/////////////////////////////////////////////////////////////////////////////////
// Main function for ParametersBase propagation 
/////////////////////////////////////////////////////////////////////////////////

const Trk::ParametersBase*  Trk::RungeKuttaPropagator::propagate
(const Trk::ParametersBase           & Tp,
 const Trk::Surface                  & Su,
 Trk::PropDirection                    D ,
 Trk::BoundaryCheck                    B ,
 const MagneticFieldProperties       & M ,
 ParticleHypothesis                    G ,
 bool                          returnCurv) const
{
  const Trk::TrackParameters  * p = dynamic_cast<const Trk::TrackParameters*>  (&Tp);
  if (p) return propagate(*p,Su,D,B,M,G,returnCurv);

  const Trk::NeutralParameters* n = dynamic_cast<const Trk::NeutralParameters*>(&Tp);
  if (n) return propagate(*n,Su,D,B,returnCurv);
  return 0;
}
      
/////////////////////////////////////////////////////////////////////////////////
// Main function for NeutralParameters propagation 
/////////////////////////////////////////////////////////////////////////////////
      
const Trk::NeutralParameters* Trk::RungeKuttaPropagator::propagate
(const Trk::NeutralParameters        & Tp,
 const Trk::Surface                  & Su,
 Trk::PropDirection                    D ,
 Trk::BoundaryCheck                    B ,
 bool                          returnCurv) const
{
  double J[25];
  return propagateStraightLine(true,Tp,Su,D,B,J,returnCurv);
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for track parameters and covariance matrix propagation
// without transport Jacobian production
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::propagate
(const Trk::TrackParameters  & Tp,
 const Trk::Surface          & Su,
 Trk::PropDirection             D,
 Trk::BoundaryCheck             B,
 const MagneticFieldProperties& M, 
 ParticleHypothesis              ,
 bool                  returnCurv) const 
{
  double J[25];
  return propagateRungeKutta(true,Tp,Su,D,B,M,J,returnCurv);
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for track parameters and covariance matrix propagation
// with transport Jacobian production
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::propagate
(const Trk::TrackParameters   & Tp ,
 const Trk::Surface&            Su ,
 Trk::PropDirection             D  ,
 Trk::BoundaryCheck             B  ,
 const MagneticFieldProperties& M  , 
 TransportJacobian           *& Jac,
 ParticleHypothesis                ,
 bool                    returnCurv) const 
{
  double J[25];

  const Trk::TrackParameters* Tpn = propagateRungeKutta(true,Tp,Su,D,B,M,J,returnCurv);
  
  if(Tpn) {
    J[24]=J[20]; J[23]=0.; J[22]=0.; J[21]=0.; J[20]=0.;
    Jac = new Trk::TransportJacobian(J);
  }
  else Jac = 0;
  return Tpn;
}

/////////////////////////////////////////////////////////////////////////////////
// Main function to finds the closest surface
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::propagate
(const TrackParameters        & Tp  ,
 std::vector<DestSurf>        & DS  ,
 PropDirection                  D   ,
 const MagneticFieldProperties& M   ,
 ParticleHypothesis                 ,
 std::vector<unsigned int>    & Sol ,
 double                       & Path,
 bool                         usePathLim,
 bool) const
{
  Sol.erase(Sol.begin(),Sol.end()); Path = 0.; if(DS.empty()) return 0;

  m_magneticFieldProperties = &M;
  m_magneticFieldTool       = 0 ;
  m_direction               = D ; 

  // Test is it measured track parameters
  //
  const Trk::MeasuredTrackParameters* 
    Mp = dynamic_cast<const Trk::MeasuredTrackParameters*>(&Tp);
  bool useJac; Mp ? useJac = true : useJac = false;


  // Magnetic field information preparation
  //
  if(useJac && m_usegradient) {
    m_magneticFieldTool = dynamic_cast<const Trk::IMagneticFieldTool*>(M.magneticFieldTool());
  }
  m_mcondition = false;
  if(m_magneticFieldProperties && m_magneticFieldProperties->magneticFieldMode()!=Trk::NoField) 
    m_mcondition = true;

  // Transform to global presentation
  //
  Trk::RungeKuttaUtils utils;

  double Po[45],Pn[45]; 
  if(!utils.transformLocalToGlobal(useJac,Tp,Po)) return 0; Po[42]=Po[43]=Po[44]=0.;

  // Straight line track propagation for small step
  //
  if(D!=0) {
    double S= m_straightStep; if(D < 0) S = -S; S = straightLineStep(useJac,S,Po);
  }

  double Wmax  = 50000.    ; // Max pass
  double W     = 0.        ; // Current pass
  double Smax  = 100.      ; // Max step 
  if(D < 0) Smax = -Smax;
  if(usePathLim) Wmax = fabs(Path);

  std::multimap<double,int> DN; double Scut[3];
  int Nveto = utils.fillDistancesMap(DS,DN,Po,W,Tp.associatedSurface(),Scut);

  // Test conditions tor start propagation and chocse direction if D == 0
  //
  if(DN.empty()) return 0;

  if(D == 0 && fabs(Scut[0]) < fabs(Scut[1])) Smax = -Smax; 

  if(Smax < 0. && Scut[0] > Smax) Smax =    Scut[0];
  if(Smax > 0. && Scut[1] < Smax) Smax =    Scut[1];
  if(Wmax >    3.*Scut[2]       ) Wmax = 3.*Scut[2]; 

  double                 Sl   = Smax ;
  double                 St   = Smax ;  
  bool                   InS  = false;
  const TrackParameters* To   = 0    ;

  for(int i=0; i!=45; ++i) Pn[i]=Po[i];
  
  //----------------------------------Niels van Eldik patch
  double last_St    =   0. ;
  bool   last_InS   = !InS ;
  bool   reverted_P = false;
  //----------------------------------

  while (fabs(W) < Wmax) {

    std::pair<double,int> SN;
    double                 S;

    if(m_mcondition) {
      
      //----------------------------------Niels van Eldik patch
      if (reverted_P && St == last_St && InS == last_InS /*&& condition_fulfiled*/) {
          // inputs are not changed will get same result.
          break;
      }
      last_St  =  St;
      last_InS = InS;
      //----------------------------------

      if(!m_magneticFieldTool) S = rungeKuttaStep            (useJac,St,Pn,InS);
      else                     S = rungeKuttaStepWithGradient(useJac,St,Pn,InS);
    }
    else  {

      //----------------------------------Niels van Eldik patch
      if (reverted_P && St == last_St /*&& !condition_fulfiled*/) {
          // inputs are not changed will get same result.
          break;
      }
      last_St  =  St;
      last_InS = InS;
      //----------------------------------

      S = straightLineStep(useJac,St,Pn);
    }
    //----------------------------------Niels van Eldik patch
    reverted_P=false;
    //----------------------------------

    bool next; SN=utils.stepEstimator(DS,DN,Po,Pn,W,m_straightStep,Nveto,next); 

    if(next) {for(int i=0; i!=45; ++i) Po[i]=Pn[i]; W+=S; Nveto=-1; }
    else     {for(int i=0; i!=45; ++i) Pn[i]=Po[i]; reverted_P=true;}

    if (fabs(S)+1. < fabs(St)) Sl=S; 
    InS ? St = 2.*S : St = S; 

    if(SN.second >= 0) {

      double Sa = fabs(SN.first);

      if(Sa > m_straightStep) {
	if(fabs(St) > Sa) St = SN.first;
      } 
      else                                {
	Path = W+SN.first;
	if((To = crossPoint(Tp,DS,Sol,Pn,SN))) return To;
	Nveto = SN.second; St = Sl;
      }
    }
  }
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for track parameters propagation without covariance matrix
// without transport Jacobian production
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::propagateParameters
(const Trk::TrackParameters  & Tp,
 const Trk::Surface          & Su, 
 Trk::PropDirection             D,
 Trk::BoundaryCheck             B,
 const MagneticFieldProperties& M, 
 ParticleHypothesis              ,
 bool                  returnCurv) const 
{
  double J[25];
  return propagateRungeKutta(false,Tp,Su,D,B,M,J,returnCurv);
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for track parameters propagation without covariance matrix
// with transport Jacobian production
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::propagateParameters
(const Trk::TrackParameters    & Tp ,
 const Trk::Surface            & Su , 
 Trk::PropDirection              D  ,
 Trk::BoundaryCheck              B  ,
 const MagneticFieldProperties&  M  , 
 TransportJacobian            *& Jac,
 ParticleHypothesis                 ,
 bool                     returnCurv) const 
{
  double J[25];
  const Trk::TrackParameters* Tpn = propagateRungeKutta   (true,Tp,Su,D,B,M,J,returnCurv);
  
  if(Tpn) {
    J[24]=J[20]; J[23]=0.; J[22]=0.; J[21]=0.; J[20]=0.;
    Jac = new Trk::TransportJacobian(J);
  }
  else Jac = 0;
  return Tpn;
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for neutral track parameters propagation with or without jacobian
/////////////////////////////////////////////////////////////////////////////////

const Trk::NeutralParameters* Trk::RungeKuttaPropagator::propagateStraightLine
(bool                           useJac,
 const Trk::NeutralParameters & Tp    ,
 const Trk::Surface           & Su    ,
 Trk::PropDirection             D     ,
 Trk::BoundaryCheck             B     ,
 double                       * Jac   ,
 bool                       returnCurv) const 
{
  const Trk::MeasuredNeutralParameters* 
    Mp = dynamic_cast<const Trk::MeasuredNeutralParameters*>(&Tp);

  if(&Su == Tp.associatedSurface()) return buildTrackParametersWithoutPropagation(Tp,Mp);

  m_magneticFieldProperties = 0    ;
  m_magneticFieldTool       = 0    ;
  m_direction               = D    ;
  m_mcondition              = false;

  Trk::RungeKuttaUtils utils;

  double P[45]; if(!utils.transformLocalToGlobal(useJac,Tp,P)) return 0;

  double Step = 0.;
  const Trk::ConeSurface*         con = 0;
  const Trk::DiscSurface*         dis = 0;
  const Trk::PlaneSurface*        pla = 0;
  const Trk::PerigeeSurface*      per = 0;
  const Trk::CylinderSurface*     cyl = 0;
  const Trk::StraightLineSurface* lin = 0;
  
  if      ((pla=dynamic_cast<const Trk::PlaneSurface*>       (&Su))) {

    const Trk::GlobalPosition&  R = pla->center(); 
    const GlobalDirection& A = pla->normal();
    double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(useJac,1,s,P,Step)) return 0;
  }
  else if ((lin=dynamic_cast<const Trk::StraightLineSurface*>(&Su))) {

    const Trk::GlobalPosition&      R = lin->center();
    const Trk::Transform3D&  T = lin->transform(); 
    double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
    if(!propagateWithJacobian(useJac,0,s,P,Step)) return 0;
  }
  else if ((dis=dynamic_cast<const Trk::DiscSurface*>        (&Su))) {

    const Trk::GlobalPosition&  R = dis->center  (); 
    const GlobalDirection& A = dis->normal  ();
    double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(useJac,1,s,P,Step)) return 0;
  }
  else if ((cyl=dynamic_cast<const Trk::CylinderSurface*>    (&Su))) {

    const Trk::Transform3D&  T = cyl->transform(); 
    double r0[3] = {P[0],P[1],P[2]};
    double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cyl->bounds().r(),(double)D,0.};
    if(!propagateWithJacobian(useJac,2,s,P,Step)) return 0;

    // For cylinder we do test for next cross point
    //
    if(cyl->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cyl,r0,P)) {
      s[8] = 0.; if(!propagateWithJacobian(useJac,2,s,P,Step)) return 0;
    }
  }
  else if ((per=dynamic_cast<const Trk::PerigeeSurface*>     (&Su))) {

    const Trk::GlobalPosition& R = per->center();  
    double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
    if(!propagateWithJacobian(useJac,0,s,P,Step)) return 0;
  }
  else if ((con=dynamic_cast<const Trk::ConeSurface*>        (&Su))) {
    
    const Trk::Transform3D&  T = con->transform(); 
    double k     = con->bounds().tanAlpha(); k = k*k+1.;
    double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),k,(double)D,0.};
    if(!propagateWithJacobian(useJac,3,s,P,Step)) return 0;
  }
  else return 0;

  if(D && (double(D)*Step)<0.) return 0;

  // Common transformation for all surfaces (angles and momentum)
  //
  if(useJac) {
    double p=1./P[6]; P[35]*=p; P[36]*=p; P[37]*=p; P[38]*=p; P[39]*=p; P[40]*=p;
  }
  double p[5]; utils.transformGlobalToLocal(useJac,P,p);

  // Surface dependent transformations (local coordinates)
  //
  bool uJ = useJac; if(returnCurv) uJ = false;
  if     (pla) {
    utils.transformGlobalToLocal(uJ,P,*pla,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!pla->insideBounds(L,0.)) return 0;}
  } 
  else if(lin) {
    utils.transformGlobalToLocal(uJ,P,*lin,p,Jac); 
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!lin->insideBounds(L,0.)) return 0;}
  }
  else if(dis) {
    utils.transformGlobalToLocal(uJ,P,*dis,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!dis->insideBounds(L,0.)) return 0;}
  } 
  else if(cyl) {
    utils.transformGlobalToLocal(uJ,P,*cyl,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!cyl->insideBounds(L,0.)) return 0;}
  } 
  else if(per) {
    utils.transformGlobalToLocal(uJ,P,*per,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!per->insideBounds(L,0.)) return 0;}
  }
  else         {
    utils.transformGlobalToLocal(uJ,P,*con,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!con->insideBounds(L,0.)) return 0;}
  }

  // Transformation to curvilinear presentation
  //
  if(returnCurv)  utils.transformGlobalToCurvilinear(useJac,P,p,Jac);
 
  if(!useJac || !Mp) {

    if(!returnCurv) {

      if(pla) return new Trk::NeutralAtaPlane       (p[0],p[1],p[2],p[3],p[4],*pla);
      if(lin) return new Trk::NeutralAtaStraightLine(p[0],p[1],p[2],p[3],p[4],*lin); 
      if(dis) return new Trk::NeutralAtaDisc        (p[0],p[1],p[2],p[3],p[4],*dis); 
      if(cyl) return new Trk::NeutralAtaCylinder    (p[0],p[1],p[2],p[3],p[4],*cyl); 
      if(per) return new Trk::NeutralPerigee        (p[0],p[1],p[2],p[3],p[4],*per);
      return 0;
    }
    else            {
      Trk::GlobalPosition gp(P[0],P[1],P[2]);
      return new Trk::NeutralCurvilinearParameters(gp,p[2],p[3],p[4]);
    }
  }

  if(!&Mp->localErrorMatrix() || !&Mp->localErrorMatrix().covariance() || Mp->localErrorMatrix().covariance().num_row()!=5) return 0; 

  Trk::CovarianceMatrix* C = utils.newCovarianceMatrix(Jac,Mp->localErrorMatrix().covariance());
  if(C->fast(1,1)<=0. || C->fast(2,2)<=0. || C->fast(3,3)<=0. || C->fast(4,4)<=0. || C->fast(5,5)<=0.) {
    delete C; return 0;
  }

  Trk::ErrorMatrix* e = new Trk::ErrorMatrix(C);

  if(!returnCurv) {

    if(pla) return new Trk::MeasuredNeutralAtaPlane       (p[0],p[1],p[2],p[3],p[4],*pla,e);
    if(lin) return new Trk::MeasuredNeutralAtaStraightLine(p[0],p[1],p[2],p[3],p[4],*lin,e); 
    if(dis) return new Trk::MeasuredNeutralAtaDisc        (p[0],p[1],p[2],p[3],p[4],*dis,e); 
    if(cyl) return new Trk::MeasuredNeutralAtaCylinder    (p[0],p[1],p[2],p[3],p[4],*cyl,e); 
    if(per) return new Trk::MeasuredNeutralPerigee        (p[0],p[1],p[2],p[3],p[4],*per,e);
    return 0;
  }
  else            {
    Trk::GlobalPosition gp(P[0],P[1],P[2]);
    return new Trk::MeasuredNeutralCurvilinearParameters(gp,p[2],p[3],p[4],e);
  }
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for charged track parameters propagation with or without jacobian
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::propagateRungeKutta
(bool                           useJac,
 const Trk::TrackParameters   & Tp    ,
 const Trk::Surface           & Su    ,
 Trk::PropDirection             D     ,
 Trk::BoundaryCheck             B     ,
 const MagneticFieldProperties& M     ,
 double                       * Jac   ,
 bool                       returnCurv) const 
{ 
  if(!&Tp || !&Su) return 0;

  m_magneticFieldProperties = &M;
  m_magneticFieldTool       = 0 ;
  m_direction               = D ; 
  
  if(useJac && m_usegradient) {
    m_magneticFieldTool = dynamic_cast<const Trk::IMagneticFieldTool*>(M.magneticFieldTool());
  }
  
  m_mcondition =false;
  if(m_magneticFieldProperties && m_magneticFieldProperties->magneticFieldMode()!=Trk::NoField) m_mcondition = true;

  const Trk::MeasuredTrackParameters* 
    Mp = dynamic_cast<const Trk::MeasuredTrackParameters*>(&Tp);

  if(&Su == Tp.associatedSurface()) return buildTrackParametersWithoutPropagation(Tp,Mp);

  Trk::RungeKuttaUtils utils;

  double P[45]; if(!utils.transformLocalToGlobal(useJac,Tp,P)) return 0;
  double Step = 0.;
  const Trk::ConeSurface*         con = 0;
  const Trk::DiscSurface*         dis = 0;
  const Trk::PlaneSurface*        pla = 0;
  const Trk::PerigeeSurface*      per = 0;
  const Trk::CylinderSurface*     cyl = 0;
  const Trk::StraightLineSurface* lin = 0;

  if      ((pla=dynamic_cast<const Trk::PlaneSurface*>       (&Su))) {

    const Trk::GlobalPosition&  R = pla->center(); 
    const GlobalDirection& A = pla->normal();
    double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(useJac,1,s,P,Step)) return 0;
  }
  else if ((lin=dynamic_cast<const Trk::StraightLineSurface*>(&Su))) {

    const Trk::GlobalPosition&      R = lin->center();
    const Trk::Transform3D&  T = lin->transform(); 
    double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
    if(!propagateWithJacobian(useJac,0,s,P,Step)) return 0;
  }
  else if ((dis=dynamic_cast<const Trk::DiscSurface*>        (&Su))) {

    const Trk::GlobalPosition&  R = dis->center  (); 
    const GlobalDirection& A = dis->normal  ();
    double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(useJac,1,s,P,Step)) return 0;
  }
  else if ((cyl=dynamic_cast<const Trk::CylinderSurface*>    (&Su))) {

    const Trk::Transform3D&  T = cyl->transform(); 
    double r0[3] = {P[0],P[1],P[2]};
    double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cyl->bounds().r(),(double)D,0.};
    if(!propagateWithJacobian(useJac,2,s,P,Step)) return 0;

    // For cylinder we do test for next cross point
    //
    if(cyl->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cyl,r0,P)) {
      s[8] = 0.; if(!propagateWithJacobian(useJac,2,s,P,Step)) return 0;
    }
  }
  else if ((per=dynamic_cast<const Trk::PerigeeSurface*>     (&Su))) {

    const Trk::GlobalPosition& R = per->center();  
    double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
    if(!propagateWithJacobian(useJac,0,s,P,Step)) return 0;
  }
  else if ((con=dynamic_cast<const Trk::ConeSurface*>        (&Su))) {

    const Trk::Transform3D&  T = con->transform(); 
    double k     = con->bounds().tanAlpha(); k = k*k+1.;
    double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),k,(double)D,0.};
    if(!propagateWithJacobian(useJac,3,s,P,Step)) return 0;
  }
 else return 0;

  if(m_direction && (m_direction*Step)<0.) {return 0;}

  // Common transformation for all surfaces (angles and momentum)
  //
  if(useJac) {
    double p=1./P[6]; P[35]*=p; P[36]*=p; P[37]*=p; P[38]*=p; P[39]*=p; P[40]*=p;
  }
  double p[5]; utils.transformGlobalToLocal(useJac,P,p);

  // Surface dependent transformations (local coordinates)
  //
  bool uJ = useJac; if(returnCurv) uJ = false;
  if     (pla) {
    utils.transformGlobalToLocal(uJ,P,*pla,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!pla->insideBounds(L,0.)) return 0;}
  } 
  else if(lin) {
    utils.transformGlobalToLocal(uJ,P,*lin,p,Jac); 
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!lin->insideBounds(L,0.)) return 0;}
  }
  else if(dis) {
    utils.transformGlobalToLocal(uJ,P,*dis,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!dis->insideBounds(L,0.)) return 0;}
  } 
  else if(cyl) {
    utils.transformGlobalToLocal(uJ,P,*cyl,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!cyl->insideBounds(L,0.)) return 0;}
  } 
  else if(per) {
    utils.transformGlobalToLocal(uJ,P,*per,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!per->insideBounds(L,0.)) return 0;}
  }
  else         {
    utils.transformGlobalToLocal(uJ,P,*con,p,Jac);
    if(B) {Trk::LocalPosition L(p[0],p[1]); if(!con->insideBounds(L,0.)) return 0;}
  }

  // Transformation to curvilinear presentation
  //
  if(returnCurv)  utils.transformGlobalToCurvilinear(useJac,P,p,Jac);
 
  if(!useJac || !Mp) {

    if(!returnCurv) {

      if(pla) return new Trk::AtaPlane       (p[0],p[1],p[2],p[3],p[4],*pla);
      if(lin) return new Trk::AtaStraightLine(p[0],p[1],p[2],p[3],p[4],*lin); 
      if(dis) return new Trk::AtaDisc        (p[0],p[1],p[2],p[3],p[4],*dis); 
      if(cyl) return new Trk::AtaCylinder    (p[0],p[1],p[2],p[3],p[4],*cyl); 
      if(per) return new Trk::Perigee        (p[0],p[1],p[2],p[3],p[4],*per);
      if(con) {Trk::GlobalPosition gp(P[0],P[1],P[2]); return new Trk::CurvilinearParameters(gp,p[2],p[3],p[4]);}
      return 0;
    } 
    else            {
      Trk::GlobalPosition gp(P[0],P[1],P[2]);
      return new Trk::CurvilinearParameters(gp,p[2],p[3],p[4]);
    }
  }

  if(!&Mp->localErrorMatrix() || !&Mp->localErrorMatrix().covariance() || Mp->localErrorMatrix().covariance().num_row()!=5) return 0; 

  Trk::CovarianceMatrix* C = utils.newCovarianceMatrix(Jac,Mp->localErrorMatrix().covariance());
  if(C->fast(1,1)<=0. || C->fast(2,2)<=0. || C->fast(3,3)<=0. || C->fast(4,4)<=0. || C->fast(5,5)<=0.) {
    delete C; return 0;
  }

  Trk::ErrorMatrix* e = new Trk::ErrorMatrix(C);

  if(!returnCurv) {

    if(pla) return new Trk::MeasuredAtaPlane       (p[0],p[1],p[2],p[3],p[4],*pla,e);
    if(lin) return new Trk::MeasuredAtaStraightLine(p[0],p[1],p[2],p[3],p[4],*lin,e); 
    if(dis) return new Trk::MeasuredAtaDisc        (p[0],p[1],p[2],p[3],p[4],*dis,e); 
    if(cyl) return new Trk::MeasuredAtaCylinder    (p[0],p[1],p[2],p[3],p[4],*cyl,e); 
    if(per) return new Trk::MeasuredPerigee        (p[0],p[1],p[2],p[3],p[4],*per,e);
    if(con) {Trk::GlobalPosition gp(P[0],P[1],P[2]); return new Trk::MeasuredCurvilinearParameters(gp,p[2],p[3],p[4],e);}
    return 0;
  }
  else            {
    Trk::GlobalPosition gp(P[0],P[1],P[2]);
    return new Trk::MeasuredCurvilinearParameters(gp,p[2],p[3],p[4],e);
  }
}

/////////////////////////////////////////////////////////////////////////////////
// Global positions calculation inside CylinderBounds
// where mS - max step allowed if mS > 0 propagate along    momentum
//                                mS < 0 propogate opposite momentum
/////////////////////////////////////////////////////////////////////////////////

void Trk::RungeKuttaPropagator::globalPositions 
(std::list<Trk::GlobalPosition>& GP,
 const TrackParameters         & Tp,
 const MagneticFieldProperties & M ,
 const CylinderBounds          & CB,
 double                          mS,
 ParticleHypothesis                ) const
{
  Trk::RungeKuttaUtils utils;
  double P[45]; if(!utils.transformLocalToGlobal(false,Tp,P)) return;

  m_direction = fabs(mS);
  if(mS > 0.) globalOneSidePositions(GP,P,M,CB, mS);
  else        globalTwoSidePositions(GP,P,M,CB,-mS);
}

/////////////////////////////////////////////////////////////////////////////////
//  Global position together with direction of the trajectory on the surface
/////////////////////////////////////////////////////////////////////////////////

const Trk::IntersectionSolution* Trk::RungeKuttaPropagator::intersect
( const Trk::TrackParameters   & Tp,
  const Trk::Surface           & Su,
  const MagneticFieldProperties& M ,
  ParticleHypothesis               ) const 
{
  bool nJ = false;
  
  m_magneticFieldProperties = &M   ;
  m_magneticFieldTool       = 0    ;
  m_direction               = 0.   ;
  m_mcondition              = false;
  if(m_magneticFieldProperties && m_magneticFieldProperties->magneticFieldMode()!=Trk::NoField) m_mcondition = true;

  Trk::RungeKuttaUtils utils;

  double P[45]; if(!utils.transformLocalToGlobal(false,Tp,P)) return 0;
  double Step = 0.;
  const Trk::DiscSurface*         dis = 0;
  const Trk::PlaneSurface*        pla = 0;
  const Trk::PerigeeSurface*      per = 0;
  const Trk::CylinderSurface*     cyl = 0;
  const Trk::StraightLineSurface* lin = 0;

  if      ((pla=dynamic_cast<const Trk::PlaneSurface*>       (&Su))) {

    const Trk::GlobalPosition&  R = pla->center(); 
    const GlobalDirection& A = pla->normal();
    double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(nJ,1,s,P,Step)) return 0;
  }
  else if ((lin=dynamic_cast<const Trk::StraightLineSurface*>(&Su))) {

    const Trk::GlobalPosition&      R = lin->center();
    const Trk::Transform3D&  T = lin->transform(); 
    double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
    if(!propagateWithJacobian(nJ,0,s,P,Step)) return 0;
  }
  else if ((dis=dynamic_cast<const Trk::DiscSurface*>        (&Su))) {

    const Trk::GlobalPosition&  R = dis->center  (); 
    const GlobalDirection& A = dis->normal  ();
    double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(nJ,1,s,P,Step)) return 0;
  }
  else if ((cyl=dynamic_cast<const Trk::CylinderSurface*>    (&Su))) {

    const Trk::Transform3D&  T = cyl->transform(); 
    double r0[3] = {P[0],P[1],P[2]};
    double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cyl->bounds().r(),1.,0.};
    if(!propagateWithJacobian(nJ,2,s,P,Step)) return 0;

    // For cylinder we do test for next cross point
    //
    if(cyl->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cyl,r0,P)) {
      s[8] = 0.; if(!propagateWithJacobian(nJ,2,s,P,Step)) return 0;
    }
  }
  else if ((per=dynamic_cast<const Trk::PerigeeSurface*>     (&Su))) {

    const Trk::GlobalPosition& R = per->center();  
    double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
    if(!propagateWithJacobian(nJ,0,s,P,Step)) return 0;
  }
  else return 0;

  Trk::GlobalPosition Glo(P[0],P[1],P[2]);
  Trk::GlobalDirection  Dir(P[3],P[4],P[5]);
  Trk::IntersectionSolution* Int = new Trk::IntersectionSolution();
  Int->push_back(new Trk::TrackSurfaceIntersection(Glo,Dir,Step));    
  return Int;
}                                                   

/////////////////////////////////////////////////////////////////////////////////
// Runge Kutta main program for propagation with or without Jacobian
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::propagateWithJacobian   
(bool Jac                         ,
 int kind                         ,
 double                       * Su,
 double                       * P ,
 double                       & W ) const
{
  const double Smax   = 1000.     ;  // max. step allowed
  double       Wmax   = 100000.   ;  // Max way allowed
  double       Wwrong = 500.      ;  // Max way with wrong direction
  double*      R      = &P[ 0]    ;  // Start coordinates
  double*      A      = &P[ 3]    ;  // Start directions
  double*      SA     = &P[42]    ; SA[0]=SA[1]=SA[2]=0.;

  if(m_mcondition && fabs(P[6]) > .1) return false; 

  // Step estimation until surface
  //
  Trk::RungeKuttaUtils utils;
  bool Q; double S, Step=utils.stepEstimator(kind,Su,P,Q); if(!Q) return false;

  bool dir = true;
  if(m_mcondition && m_direction && m_direction*Step < 0.)  {
    Step = -Step; dir = false;
  }

  Step>Smax ? S=Smax : Step<-Smax ? S=-Smax : S=Step;
  double So = fabs(S); int iS = 0;

  bool InS = false;

  // Rkuta extrapolation
  //
  int niter = 0;
  while(fabs(Step) > m_straightStep) {

    if(++niter > 10000) return false;

    if(m_mcondition) {

      if(!m_magneticFieldTool) W+=(S=rungeKuttaStep            (Jac,S,P,InS));
      else                     W+=(S=rungeKuttaStepWithGradient(Jac,S,P,InS));
    }
    else  {
      W+=(S=straightLineStep(Jac,S,P));
    }

    Step = stepEstimatorWithCurvature(kind,Su,P,Q); if(!Q) return false;

    if(!dir) {
      if(m_direction && m_direction*Step < 0.)  Step = -Step;
      else                                      dir  =  true;
    }
    
    if(S*Step<0.) {S = -S; ++iS;}

    double aS    = fabs(S   );
    double aStep = fabs(Step);
    if     (    aS > aStep             )  S = Step;
    else if(!iS && InS && aS*2. < aStep)  S*=2.   ;
    if(!dir && fabs(W) > Wwrong              ) return false;
    if(iS > 10 || (iS>3 && fabs(S)>=So) || fabs(W)>Wmax ) {if(!kind) break; return false;}
    So=fabs(S);
  }
  
  // Output track parameteres
  //
  W+=Step;

  if(fabs(Step) < .001) return true;

  A [0]+=(SA[0]*Step); 
  A [1]+=(SA[1]*Step);
  A [2]+=(SA[2]*Step);
  double CBA  = 1./sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
  R[0]+=Step*(A[0]-.5*Step*SA[0]); A[0]*=CBA;
  R[1]+=Step*(A[1]-.5*Step*SA[1]); A[1]*=CBA;
  R[2]+=Step*(A[2]-.5*Step*SA[2]); A[2]*=CBA;
  return true;
}

/////////////////////////////////////////////////////////////////////////////////
// Runge Kutta trajectory model (units->mm,MeV,kGauss)
// Uses Nystroem algorithm (See Handbook Net. Bur. ofStandards, procedure 25.5.20)
/////////////////////////////////////////////////////////////////////////////////

double Trk::RungeKuttaPropagator::rungeKuttaStep
  (bool                           Jac,
   double                         S  , 
   double                       * P  ,
   bool                         & InS) const
 {
  double* R    =          &P[ 0];            // Coordinates 
  double* A    =          &P[ 3];            // Directions
  double* sA   =          &P[42];
  double  Pi   = .014989626*P[6];            // Invert mometum/2. 
  double  dltm = m_dlt*.03      ;

  CLHEP::Hep3Vector f0,f; Trk::GlobalPosition gP(R[0],R[1],R[2]);  
  m_magneticFieldProperties->getMagneticFieldKiloGauss(gP,f0);

  bool Helix = false; if(fabs(S) < m_helixStep) Helix = true; 
  while(fabs(S) > 0.) {
 
    
    double S3=.33333333*S, S4=.25*S, PS2=Pi*S;
    __attribute__((aligned(32))) double S3_V[3] = {S3, S3, S3};

    // First point
    //   
    __attribute__((aligned(32))) double H0[3] = {f0.x()*PS2,f0.y()*PS2,f0.z()*PS2};
    double A0    = A[1]*H0[2]-A[2]*H0[1]             ;
    double B0    = A[2]*H0[0]-A[0]*H0[2]             ;
    double C0    = A[0]*H0[1]-A[1]*H0[0]             ;
    __attribute__((aligned(32))) double V0[3] = {A0, B0, C0};
    double A2    = A[0]+A0                           ;
    double B2    = A[1]+B0                           ;
    double C2    = A[2]+C0                           ;
    double A1    = A2+A[0]                           ;
    double B1    = B2+A[1]                           ;
    double C1    = C2+A[2]                           ;
    
    // Second point
    //
    if(!Helix) {
      gP[0]=R[0]+A1*S4; gP[1]=R[1]+B1*S4; gP[2]=R[2]+C1*S4; 
      m_magneticFieldProperties->getMagneticFieldKiloGauss(gP,f);
    }
    else       {f[0]=f0.x(); f[1]=f0.y(); f[2]=f0.z();}

    __attribute__((aligned(32))) double H1[3] = {f.x()*PS2,f.y()*PS2,f.z()*PS2}; 
    double A3    = B2*H1[2]-C2*H1[1]+A[0]         ; 
    double B3    = C2*H1[0]-A2*H1[2]+A[1]         ; 
    double C3    = A2*H1[1]-B2*H1[0]+A[2]         ;
    __attribute__((aligned(32))) double V3[3] = {A3, B3, C3};
    double A4    = B3*H1[2]-C3*H1[1]+A[0]         ; 
    double B4    = C3*H1[0]-A3*H1[2]+A[1]         ; 
    double C4    = A3*H1[1]-B3*H1[0]+A[2]         ;
    __attribute__((aligned(32))) double V4[3] = {A4, B4, C4};
    double A5    = A4-A[0]+A4                     ; 
    double B5    = B4-A[1]+B4                     ; 
    double C5    = C4-A[2]+C4                     ;
    
    // Last point
    //
    if(!Helix) {
      gP[0]=R[0]+S*A4; gP[1]=R[1]+S*B4; gP[2]=R[2]+S*C4;    
      m_magneticFieldProperties->getMagneticFieldKiloGauss(gP,f);
    }
    else       {f[0]=f0.x(); f[1]=f0.y(); f[2]=f0.z();} 

    __attribute__((aligned(32))) double H2[3] = {f.x()*PS2,f.y()*PS2,f.z()*PS2}; 
    double A6    = B5*H2[2]-C5*H2[1]              ;
    double B6    = C5*H2[0]-A5*H2[2]              ;
    double C6    = A5*H2[1]-B5*H2[0]              ;
    __attribute__((aligned(32))) double V6[3] = {A6, B6, C6};
    
    /*
    // Test approximation quality on give step and possible step reduction
    //
    double dE[4] = {A1+A6-A3-A4,B1+B6-B3-B4,C1+C6-C3-C4,S};
    double cS    = stepReduction(dE);
    if     (cSn <  1.) {S*=cS; continue;}
    cSn == 1. InS = false : InS = true;
    */

    // Test approximation quality on give step and possible step reduction
    //
    double EST = fabs(A1+A6-A3-A4)+fabs(B1+B6-B3-B4)+fabs(C1+C6-C3-C4); 
    if(EST>m_dlt) {S*=.5; continue;} EST<dltm ? InS = true : InS = false;

    // Parameters calculation
    //   
    double Sl = S; if(Sl!=0.) Sl = 1./Sl;
    double A00 = A[0], A11=A[1], A22=A[2];
    __attribute__((aligned(32))) double A_V[3] = {A00, A11, A22};
    R[0]+=(A2+A3+A4)*S3; A[0]+=(sA[0]=(A0+A3+A3+A5+A6)*.33333333-A[0]); sA[0]*=Sl; 
    R[1]+=(B2+B3+B4)*S3; A[1]+=(sA[1]=(B0+B3+B3+B5+B6)*.33333333-A[1]); sA[1]*=Sl;
    R[2]+=(C2+C3+C4)*S3; A[2]+=(sA[2]=(C0+C3+C3+C5+C6)*.33333333-A[2]); sA[2]*=Sl;
    double CBA = 1./sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
    A[0]*=CBA; A[1]*=CBA; A[2]*=CBA;

    if(!Jac) return S;

    // Jacobian calculation
    __m256d C_012 = _mm256_set1_pd(0.33333333);
    __m256d A_012 = _mm256_load_pd(A_V);
    __m256d S3_012 = _mm256_load_pd(S3_V);
    __m256d V0_012 = _mm256_load_pd(V0);
    __m256d V3_012 = _mm256_load_pd(V3);
    __m256d V4_012 = _mm256_load_pd(V4);
    __m256d V6_012 = _mm256_load_pd(V6);

    __m256d H0_012 = _mm256_load_pd(H0);
    __m256d H0_201 = CROSS_SHUFFLE_201(H0_012);
    __m256d H0_120 = CROSS_SHUFFLE_120(H0_012);

    __m256d H1_012 = _mm256_load_pd(H1);
    __m256d H1_201 = CROSS_SHUFFLE_201(H1_012);
    __m256d H1_120 = CROSS_SHUFFLE_120(H1_012);

    __m256d H2_012 = _mm256_load_pd(H1);
    __m256d H2_201 = CROSS_SHUFFLE_201(H2_012);
    __m256d H2_120 = CROSS_SHUFFLE_120(H2_012);

    for(int i = 7; i < 42; i+=7){
      __m256d dR = _mm256_loadu_pd(&P[i]);
    
      __m256d dA = _mm256_loadu_pd(&P[i + 3]);
      __m256d dA_201 = CROSS_SHUFFLE_201(dA);
      __m256d dA_120 = CROSS_SHUFFLE_120(dA);

      __m256d d0 = _mm256_sub_pd(_mm256_mul_pd(H0_201, dA_120), _mm256_mul_pd(H0_120, dA_201));

      if(i==35){
	d0 = _mm256_add_pd(d0, V0_012);
      }
    
      __m256d d2 = _mm256_add_pd(d0, dA);
      __m256d d2_201 = CROSS_SHUFFLE_201(d2);
      __m256d d2_120 = CROSS_SHUFFLE_120(d2);
      
      __m256d d3 = _mm256_sub_pd(_mm256_add_pd(dA, _mm256_mul_pd(d2_120, H1_201)), _mm256_mul_pd(d2_201, H1_120));
      if(i==35){
	d3 = _mm256_add_pd(d3, _mm256_sub_pd(V3_012, A_012));
      }
      __m256d d3_201 = CROSS_SHUFFLE_201(d3);
      __m256d d3_120 = CROSS_SHUFFLE_120(d3);


      __m256d d4 = _mm256_sub_pd(_mm256_add_pd(dA, _mm256_mul_pd(d3_120, H1_201)), _mm256_mul_pd(d3_201, H1_120));

      if(i==35){
        d4 = _mm256_add_pd(d4, _mm256_sub_pd(V4_012, A_012));
      }

      __m256d d5 = _mm256_sub_pd(_mm256_add_pd(d4, d4), dA);
      __m256d d5_201 = CROSS_SHUFFLE_201(d5);
      __m256d d5_120 = CROSS_SHUFFLE_120(d5);

      __m256d d6 = _mm256_sub_pd(_mm256_mul_pd(d5_120, H2_201), _mm256_mul_pd(d5_201, H2_120));

      if(i==35){
        d6 = _mm256_add_pd(d6, V6_012);
      }

      _mm256_storeu_pd(&P[i], _mm256_add_pd(dR, _mm256_mul_pd(_mm256_add_pd(d2, _mm256_add_pd(d3, d4)), S3_012)));
      _mm256_storeu_pd(&P[i + 3], _mm256_mul_pd(C_012, _mm256_add_pd(d0, _mm256_add_pd(d3, _mm256_add_pd(d3, _mm256_add_pd(d5, d6))))));
    }

    return S;
  }
  return S;
}

/////////////////////////////////////////////////////////////////////////////////
// Runge Kutta trajectory model (units->mm,MeV,kGauss)
// Uses Nystroem algorithm (See Handbook Net. Bur. ofStandards, procedure 25.5.20)
//    Where magnetic field information iS              
//    f[ 0],f[ 1],f[ 2] - Hx    , Hy     and Hz of the magnetic field         
//    f[ 3],f[ 4],f[ 5] - dHx/dx, dHx/dy and dHx/dz                           
//    f[ 6],f[ 7],f[ 8] - dHy/dx, dHy/dy and dHy/dz                           
//    f[ 9],f[10],f[11] - dHz/dx, dHz/dy and dHz/dz                           
//                                                                                   
/////////////////////////////////////////////////////////////////////////////////

double Trk::RungeKuttaPropagator::rungeKuttaStepWithGradient
(bool                           Jac,
 double                         S  , 
 double                       * P  ,
 bool                         & InS) const
{
  double* R    =          &P[ 0];           // Coordinates 
  double* A    =          &P[ 3];           // Directions
  double* sA   =          &P[42];
  double  Pi   = .014989626*P[6];           // Invert mometum/2. 
  double  dltm = m_dlt*.03      ;

  float  f0[12],f1[12],f2[12];
  double H0[12],H1[12],H2[12];
  float pos[3] = {R[0],R[1],R[2]};
  m_magneticFieldTool->magFieldAthena()->fieldGradient_XYZ_in_mm(pos,f0);

  bool Helix = false; if(fabs(S) < m_helixStep) Helix = true; 
  while(fabs(S) > 0.) {
 
    
    double S3=.33333333*S, S4=.25*S, PS2=Pi*S;

    // First point
    //   
    H0[0] = f0[0]*PS2; H0[1] = f0[1]*PS2; H0[2] = f0[2]*PS2;
    double A0    = A[1]*H0[2]-A[2]*H0[1]             ;
    double B0    = A[2]*H0[0]-A[0]*H0[2]             ;
    double C0    = A[0]*H0[1]-A[1]*H0[0]             ;
    double A2    = A[0]+A0                           ;
    double B2    = A[1]+B0                           ;
    double C2    = A[2]+C0                           ;
    double A1    = A2+A[0]                           ;
    double B1    = B2+A[1]                           ;
    double C1    = C2+A[2]                           ;
    
    // Second point
    //
    if(!Helix) {
      pos[0]=R[0]+A1*S4; pos[1]=R[1]+B1*S4; pos[2]=R[2]+C1*S4; 
      m_magneticFieldTool->magFieldAthena()->fieldGradient_XYZ_in_mm(pos,f1);
    }
    else       {f1[0]=f0[0]; f1[1]=f0[1]; f1[2]=f0[2];}

    H1[0] = f1[0]*PS2; H1[1] = f1[1]*PS2; H1[2] = f1[2]*PS2; 
    double A3    = B2*H1[2]-C2*H1[1]+A[0]         ; 
    double B3    = C2*H1[0]-A2*H1[2]+A[1]         ; 
    double C3    = A2*H1[1]-B2*H1[0]+A[2]         ;
    double A4    = B3*H1[2]-C3*H1[1]+A[0]         ; 
    double B4    = C3*H1[0]-A3*H1[2]+A[1]         ; 
    double C4    = A3*H1[1]-B3*H1[0]+A[2]         ;
    double A5    = A4-A[0]+A4                     ; 
    double B5    = B4-A[1]+B4                     ; 
    double C5    = C4-A[2]+C4                     ;
    
    // Last point
    //
    if(!Helix) {
      pos[0]=R[0]+S*A4; pos[1]=R[1]+S*B4; pos[2]=R[2]+S*C4;    
      m_magneticFieldTool->magFieldAthena()->fieldGradient_XYZ_in_mm(pos,f2);
    }
    else       {f2[0]=f0[0]; f2[1]=f0[1]; f2[2]=f0[2];} 

    H2[0] = f2[0]*PS2; H2[1] = f2[1]*PS2; H2[2] = f2[2]*PS2; 
    double A6    = B5*H2[2]-C5*H2[1]              ;
    double B6    = C5*H2[0]-A5*H2[2]              ;
    double C6    = A5*H2[1]-B5*H2[0]              ;

    /*
    // Test approximation quality on give step and possible step reduction
    //
    double dE[4] = {A1+A6-A3-A4,B1+B6-B3-B4,C1+C6-C3-C4,S};
    double cS    = stepReduction(dE);
    if     (cSn <  1.) {S*=cS; continue;}
    cSn == 1. InS = false : InS = true;
    */
    
    // Test approximation quality on give step and possible step reduction
    //
    double EST = fabs(A1+A6-A3-A4)+fabs(B1+B6-B3-B4)+fabs(C1+C6-C3-C4); 
    if(EST>m_dlt) {S*=.5; continue;} EST<dltm ? InS = true : InS = false;

    // Parameters calculation
    //   
    double Sl = S; if(Sl!=0.) Sl = 1./Sl;
    double A00 = A[0], A11=A[1], A22=A[2];
    R[0]+=(A2+A3+A4)*S3; A[0]+=(sA[0]=(A0+A3+A3+A5+A6)*.33333333-A[0]); sA[0]*=Sl; 
    R[1]+=(B2+B3+B4)*S3; A[1]+=(sA[1]=(B0+B3+B3+B5+B6)*.33333333-A[1]); sA[1]*=Sl;
    R[2]+=(C2+C3+C4)*S3; A[2]+=(sA[2]=(C0+C3+C3+C5+C6)*.33333333-A[2]); sA[2]*=Sl;
    double CBA = 1./sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
    A[0]*=CBA; A[1]*=CBA; A[2]*=CBA;

    if(!Jac) return S;

    // Jacobian calculation
    //
    if(!Helix) {
      for(int i=3; i!=12; ++i) {H0[i]=f0[i]*PS2; H1[i]=f1[i]*PS2; H2[i]=f2[i]*PS2;}
    }
    else       {
      for(int i=3; i!=12; ++i) {H0[i]=f0[i]*PS2; H1[i]=H0[i];     H2[i]=H0[i];    }
    }

    for(int i=7; i<42; i+=7) {

      double* dR   = &P[i]                                          ;  
      double* dA   = &P[i+3]                                        ;
      
      double dH0   = H0[ 3]*dR[0]+H0[ 4]*dR[1]+H0[ 5]*dR[2]         ; // dHx/dp
      double dH1   = H0[ 6]*dR[0]+H0[ 7]*dR[1]+H0[ 8]*dR[2]         ; // dHy/dp
      double dH2   = H0[ 9]*dR[0]+H0[10]*dR[1]+H0[11]*dR[2]         ; // dHz/dp

      double dA0   = H0[ 2]*dA[1]-H0[ 1]*dA[2]+A[1]*dH2-A[2]*dH1    ; // dA0/dp
      double dB0   = H0[ 0]*dA[2]-H0[ 2]*dA[0]+A[2]*dH0-A[0]*dH2    ; // dB0/dp
      double dC0   = H0[ 1]*dA[0]-H0[ 0]*dA[1]+A[0]*dH1-A[1]*dH0    ; // dC0/dp

      if(i==35) {dA0+=A0; dB0+=B0; dC0+=C0;}

      double dA2   = dA0+dA[0], dX = dR[0]+(dA2+dA[0])*S4           ; // dX /dp
      double dB2   = dB0+dA[1], dY = dR[1]+(dB2+dA[1])*S4           ; // dY /dp
      double dC2   = dC0+dA[2], dZ = dR[2]+(dC2+dA[2])*S4           ; // dZ /dp
      dH0          = H1[ 3]*dX   +H1[ 4]*dY   +H1[ 5]*dZ            ; // dHx/dp
      dH1          = H1[ 6]*dX   +H1[ 7]*dY   +H1[ 8]*dZ            ; // dHy/dp
      dH2          = H1[ 9]*dX   +H1[10]*dY   +H1[11]*dZ            ; // dHz/dp
      double dA3   = dA[0]+dB2*H1[2]-dC2*H1[1]+B2*dH2-C2*dH1        ; // dA3/dp
      double dB3   = dA[1]+dC2*H1[0]-dA2*H1[2]+C2*dH0-A2*dH2        ; // dB3/dp
      double dC3   = dA[2]+dA2*H1[1]-dB2*H1[0]+A2*dH1-B2*dH0        ; // dC3/dp

      if(i==35) {dA3+=A3-A00; dB3+=B3-A11; dC3+=C3-A22;}

      double dA4   = dA[0]+dB3*H1[2]-dC3*H1[1]+B3*dH2-C3*dH1        ; // dA4/dp
      double dB4   = dA[1]+dC3*H1[0]-dA3*H1[2]+C3*dH0-A3*dH2        ; // dB4/dp
      double dC4   = dA[2]+dA3*H1[1]-dB3*H1[0]+A3*dH1-B3*dH0        ; // dC4/dp

      if(i==35) {dA4+=A4-A00; dB4+=B4-A11; dC4+=C4-A22;}

      double dA5   = dA4+dA4-dA[0];  dX = dR[0]+dA4*S               ; // dX /dp 
      double dB5   = dB4+dB4-dA[1];  dY = dR[1]+dB4*S               ; // dY /dp
      double dC5   = dC4+dC4-dA[2];  dZ = dR[2]+dC4*S               ; // dZ /dp
      dH0          = H2[ 3]*dX   +H2[ 4]*dY   +H2[ 5]*dZ            ; // dHx/dp
      dH1          = H2[ 6]*dX   +H2[ 7]*dY   +H2[ 8]*dZ            ; // dHy/dp
      dH2          = H2[ 9]*dX   +H2[10]*dY   +H2[11]*dZ            ; // dHz/dp
      double dA6   = dB5*H2[2]-dC5*H2[1]+B5*dH2-C5*dH1              ; // dA6/dp
      double dB6   = dC5*H2[0]-dA5*H2[2]+C5*dH0-A5*dH2              ; // dB6/dp
      double dC6   = dA5*H2[1]-dB5*H2[0]+A5*dH1-B5*dH0              ; // dC6/dp
 
      if(i==35) {dA6+=A6; dB6+=B6; dC6+=C6;}
 
      dR[0]+=(dA2+dA3+dA4)*S3; dA[0]=(dA0+dA3+dA3+dA5+dA6)*.33333333;      
      dR[1]+=(dB2+dB3+dB4)*S3; dA[1]=(dB0+dB3+dB3+dB5+dB6)*.33333333; 
      dR[2]+=(dC2+dC3+dC4)*S3; dA[2]=(dC0+dC3+dC3+dC5+dC6)*.33333333;
    }
    return S;
  }
  return S;
}

/////////////////////////////////////////////////////////////////////////////////
// Straight line trajectory model 
/////////////////////////////////////////////////////////////////////////////////

double Trk::RungeKuttaPropagator::straightLineStep
(bool    Jac,
 double  S  , 
 double* P  ) const
{
  double*  R   = &P[ 0];             // Start coordinates
  double*  A   = &P[ 3];             // Start directions
  double* sA   = &P[42];           

  // Track parameters in last point
  //
  R[0]+=(A[0]*S); R[1]+=(A[1]*S); R[2]+=(A[2]*S); if(!Jac) return S;
  
  // Derivatives of track parameters in last point
  //
  for(int i=7; i<42; i+=7) {

    double* dR = &P[i]; 
    double* dA = &P[i+3];
    dR[0]+=(dA[0]*S); dR[1]+=(dA[1]*S); dR[2]+=(dA[2]*S);
  }
  sA[0]=sA[1]=sA[2]=0.; return S;
}

/////////////////////////////////////////////////////////////////////////////////
// Test quality Jacobian calculation 
/////////////////////////////////////////////////////////////////////////////////

void Trk::RungeKuttaPropagator::JacobianTest
(const Trk::TrackParameters        & Tp,
 const Trk::Surface                & Su,
 const Trk::MagneticFieldProperties& M ) const
{
  std::cout<<"          |-------------|----------------------------------|--------------|"
	   <<std::endl;
  bool nJ = true;
  bool useJac = true;
  m_magneticFieldProperties = &M;
  m_direction = 0.;                      

  if(m_usegradient) {
    m_magneticFieldTool = dynamic_cast<const Trk::IMagneticFieldTool*>(M.magneticFieldTool());
  }

  m_mcondition = false;
  if(m_magneticFieldProperties && m_magneticFieldProperties->magneticFieldMode()!=Trk::NoField) m_mcondition = true;

  Trk::RungeKuttaUtils utils;
  double P[45]; if(!utils.transformLocalToGlobal(useJac,Tp,P)) return;
  double Step = 0.;
  const Trk::DiscSurface*         disc     = 0;
  const Trk::PlaneSurface*        plane    = 0;
  const Trk::PerigeeSurface*      perigee  = 0;
  const Trk::CylinderSurface*     cylinder = 0;
  const Trk::StraightLineSurface* line     = 0;

  if      ((plane=dynamic_cast<const Trk::PlaneSurface*>      (&Su))) {

    const Trk::GlobalPosition&  R = plane->center(); 
    const GlobalDirection& A = plane->normal();
    double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(nJ,1,s,P,Step)) return;
  }
  else if ((line=dynamic_cast<const Trk::StraightLineSurface*>(&Su))) {

    const Trk::GlobalPosition&      R = line->center();
    const Trk::Transform3D&  T = line->transform(); 
    double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
    if(!propagateWithJacobian(nJ,0,s,P,Step)) return;
  }
  else if ((disc=dynamic_cast<const Trk::DiscSurface*>        (&Su))) {

    const Trk::GlobalPosition&  R = disc->center  (); 
    const GlobalDirection& A = disc->normal  ();
    double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(nJ,1,s,P,Step)) return;
  }
  else if ((cylinder=dynamic_cast<const Trk::CylinderSurface*>(&Su))) {

    const Trk::Transform3D&  T = cylinder->transform(); 
    double r0[3] = {P[0],P[1],P[2]};
    double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cylinder->bounds().r(),1.,0.};
    if(!propagateWithJacobian(nJ,2,s,P,Step)) return;

    // For cylinder we do test for next cross point
    //
    if(cylinder->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cylinder,r0,P)) {
      s[8] = 0.; if(!propagateWithJacobian(nJ,2,s,P,Step)) return;
    }
  }
  else if ((perigee=dynamic_cast<const Trk::PerigeeSurface*>  (&Su))) {

    const Trk::GlobalPosition& R = perigee->center();  
    double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
    if(!propagateWithJacobian(nJ,0,s,P,Step)) return;
  }
  else return;

  // Common transformation for all surfaces (angles and momentum)
  //
  if(useJac) {
    double p=1./P[6]; P[35]*=p; P[36]*=p; P[37]*=p; P[38]*=p; P[39]*=p; P[40]*=p;
  }
  double par[5],JacA[21]; utils.transformGlobalToLocal(useJac,P,par);

  // Surface dependent transformations (local coordinates)
  //
  if     (plane   ) utils.transformGlobalToLocal(useJac,P,*plane   ,par,JacA); 
  else if(line    ) utils.transformGlobalToLocal(useJac,P,*line    ,par,JacA); 
  else if(disc    ) utils.transformGlobalToLocal(useJac,P,*disc    ,par,JacA); 
  else if(cylinder) utils.transformGlobalToLocal(useJac,P,*cylinder,par,JacA); 
  else              utils.transformGlobalToLocal(useJac,P,*perigee ,par,JacA); 
  
  // Numeric jacobian calculation
  //
  double dP[5]={.001,.001,.00001,.000001,.00001}, JacN[25],Jac[25]; 
  nJ = false;  useJac = false;
  m_magneticFieldTool = 0    ;

  for(int i=0; i!=5; ++i) { 
 
    transformLocalToGlobal(i,dP[i],Tp,P);
    
    if      (plane   ) {

      const Trk::GlobalPosition&  R = plane->center(); 
      const GlobalDirection& A = plane->normal();
      double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
      double s[4];
      if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
      else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
      if(!propagateWithJacobian(nJ,1,s,P,Step)) return;
    }
    else if (line    ) {

      const Trk::GlobalPosition&      R = line->center();
      const Trk::Transform3D&  T = line->transform(); 
      double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
      if(!propagateWithJacobian(nJ,0,s,P,Step)) return;
    }
    else if (disc    ) {

      const Trk::GlobalPosition&  R = disc->center  (); 
      const GlobalDirection& A = disc->normal  ();
      double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
      double s[4];
      if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
      else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
      if(!propagateWithJacobian(nJ,1,s,P,Step)) return;
    }
    else if (cylinder) {

      const Trk::Transform3D&  T = cylinder->transform(); 
      double r0[3] = {P[0],P[1],P[2]};
      double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cylinder->bounds().r(),1.,0.};
      if(!propagateWithJacobian(nJ,2,s,P,Step)) return;

      // For cylinder we do test for next cross point
      //
      if(cylinder->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cylinder,r0,P)) {
	s[8] = 0.; if(!propagateWithJacobian(nJ,2,s,P,Step)) return;
      }
    }
    else if (perigee ) {

      const Trk::GlobalPosition& R = perigee->center();  
      double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
      if(!propagateWithJacobian(nJ,0,s,P,Step)) return;
    }

    // Common transformation for all surfaces (angles and momentum)
    //
    if(useJac) {
      double p=1./P[6]; P[35]*=p; P[36]*=p; P[37]*=p; P[38]*=p; P[39]*=p; P[40]*=p;
    }
    double par1[5]; utils.transformGlobalToLocal(useJac,P,par1);
    
    // Surface dependent transformations (local coordinates)
    //
    if     (plane   ) utils.transformGlobalToLocal(useJac,P,*plane   ,par1,Jac);
    else if(line    ) utils.transformGlobalToLocal(useJac,P,*line    ,par1,Jac); 
    else if(disc    ) utils.transformGlobalToLocal(useJac,P,*disc    ,par1,Jac);
    else if(cylinder) utils.transformGlobalToLocal(useJac,P,*cylinder,par1,Jac); 
    else              utils.transformGlobalToLocal(useJac,P,*perigee ,par1,Jac); 
    JacN[i   ] = (par1[0]-par[0])/dP[i]; if(fabs(JacN[i   ])<.000001) JacN[i   ]=0.;
    JacN[i+ 5] = (par1[1]-par[1])/dP[i]; if(fabs(JacN[i+ 5])<.000001) JacN[i+ 5]=0.;
    JacN[i+10] = (par1[2]-par[2])/dP[i]; if(fabs(JacN[i+10])<.000001) JacN[i+10]=0.;
    JacN[i+15] = (par1[3]-par[3])/dP[i]; if(fabs(JacN[i+15])<.000001) JacN[i+15]=0.;
    JacN[i+20] = (par1[4]-par[4])/dP[i]; if(fabs(JacN[i+20])<.000001) JacN[i+20]=0.;

  }  
  if     (plane   ) std::cout<<"=====>|        Plane |"<<std::endl;
  else if(line    ) std::cout<<"=====>| StraightLine |"<<std::endl;
  else if(disc    ) std::cout<<"=====>|         Disc |"<<std::endl;
  else if(cylinder) std::cout<<"=====>|      Cyliner |"<<std::endl;
  else              std::cout<<"=====>|      Perigee |"<<std::endl;

  for(int i=0; i!=21; ++i) {if(fabs(JacA[i])<.00001) JacA[i]=0.;}

  if(M.magneticFieldMode()!=Trk::NoField) {
    std::cout<<"          |             |       with magnetic field        |              |"
	     <<std::endl;
  }
  else {
    std::cout<<"          |             |     without  magnetic field      |              |"
	     <<std::endl;
  }
  std::cout<<"          |-------------|----------------------------------|--------------|"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
 	   <<std::endl;
  std::cout<<"|  Parameters |         L1  |         L2  |        Phi  |        The  |        CM   |"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
 	   <<std::endl;
  std::cout<<"|  Old        |"
	   <<std::setw(12)<<std::setprecision(5)<<Tp.parameters()[0]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<Tp.parameters()[1]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<Tp.parameters()[2]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<Tp.parameters()[3]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<Tp.parameters()[4]<<" |"
	   <<std::endl;
  std::cout<<"|  New        |"
	   <<std::setw(12)<<std::setprecision(5)<<par[0]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<par[1]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<par[2]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<par[3]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<par[4]<<" |"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
	   <<std::endl;
  std::cout<<"| Jacobian(A) | Old   /dL1  |       /dL2  |      /dPhi  |      /dThe  |       /dCM  |"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
	   <<std::endl;
  std::cout<<"|  New  dL1 / |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 0]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 1]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 2]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 3]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 4]<<" |"
	   <<std::endl;
  std::cout<<"|       dL2 / |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 5]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 6]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 7]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 8]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[ 9]<<" |"
	   <<std::endl;
  std::cout<<"|       dPhi/ |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[10]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[11]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[12]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[13]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[14]<<" |"
	   <<std::endl;
  std::cout<<"|       dThe/ |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[15]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[16]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[17]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[18]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[19]<<" |"
	   <<std::endl;
  std::cout<<"|       dCM / |"
	   <<std::setw(12)<<std::setprecision(5)<<0.<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<0.<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<0.<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<0.<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacA[20]<<" |"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
	   <<std::endl;
  std::cout<<"| Jacobian(N) | Old   /dL1  |       /dL2  |      /dPhi  |      /dThe  |       /dCM  |"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
	   <<std::endl;
  std::cout<<"|  New  dL1 / |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 0]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 1]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 2]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 3]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 4]<<" |"
	   <<std::endl;
  std::cout<<"|       dL2 / |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 5]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 6]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 7]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 8]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[ 9]<<" |"
	   <<std::endl;
  std::cout<<"|       dPhi/ |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[10]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[11]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[12]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[13]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[14]<<" |"
	   <<std::endl;
  std::cout<<"|       dThe/ |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[15]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[16]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[17]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[18]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[19]<<" |"
	   <<std::endl;
  std::cout<<"|       dCM / |"<<std::setw(12)
	   <<std::setprecision(5)<<JacN[20]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[21]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[22]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[23]<<" |"
	   <<std::setw(12)<<std::setprecision(5)<<JacN[24]<<" |"
	   <<std::endl;
  std::cout<<"|-------------|-------------|-------------|-------------|-------------|-------------|"
	   <<std::endl;
}

/////////////////////////////////////////////////////////////////////////////////
// Common transformation from local to global system coordinates for all surfaces
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::transformLocalToGlobal 
(int                         i ,
 double                      dP,
 const Trk::TrackParameters& Tp,
 double                    * P ) const 
{
  double p[5] = {Tp.parameters()[0],
		 Tp.parameters()[1],
		 Tp.parameters()[2],
		 Tp.parameters()[3],
		 Tp.parameters()[4]};
  p[i]+=dP;
  
  double   Sf = sin(p[2]), Cf = cos(p[2]);
  double   Ce = cos(p[3]), Se = sqrt((1.-Ce)*(1.+Ce)); 
  P[ 3] = Cf*Se;                                                         // Ax
  P[ 4] = Sf*Se;                                                         // Ay
  P[ 5] = Ce;                                                            // Az
  P[ 6] = p[4];                                                          // CM
  if(fabs(P[6])<.000000000000001) {
    P[6]<0. ? P[6]=-.000000000000001 : P[6]= .000000000000001;
  }    
  
  Trk::RungeKuttaUtils utils;

  const Trk::Surface            * su       = Tp.associatedSurface();
  const Trk::PlaneSurface       * plane    = 0;
  const Trk::StraightLineSurface* line     = 0;
  const Trk::DiscSurface        * disc     = 0;
  const Trk::CylinderSurface    * cylinder = 0;
  const Trk::PerigeeSurface     * pline    = 0; 
  
  if     ((plane   = dynamic_cast<const Trk::PlaneSurface*>       (su))) {
    if(i==4) std::cout<<"          | Plane       |====== RungeKuttaPropagator ";
    utils.transformLocalToGlobal(false,*plane   ,p,P); return true;
  }
  else if((line    = dynamic_cast<const Trk::StraightLineSurface*>(su))) {
    if(i==4) std::cout<<"          | StraightLine|====== RungeKuttaPropagator ";
    utils.transformLocalToGlobal(false,*line    ,p,P); return true;
  }
  else if((disc    = dynamic_cast<const Trk::DiscSurface*>        (su))) {
    if(i==4) std::cout<<"          | Disc        |====== RungeKuttaPropagator ";
   utils.transformLocalToGlobal(false,*disc    ,p,P); return true;
  }
  else if((cylinder= dynamic_cast<const Trk::CylinderSurface*>    (su))) {
    if(i==4) std::cout<<"          | Cylinder    |====== RungeKuttaPropagator ";
    utils.transformLocalToGlobal(false,*cylinder,p,P); return true;
  }
  else if((pline   = dynamic_cast<const Trk::PerigeeSurface*>     (su))) {
    if(i==4) std::cout<<"          | Perigee     |====== RungeKuttaPropagator ";
    utils.transformLocalToGlobal(false,*pline,   p,P); return true;
  }
  else return false;
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for simple track parameters and covariance matrix propagation
// Ta->Su = Tb
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::propagate
(Trk::PatternTrackParameters  & Ta,
 const Trk::Surface           & Su,
 Trk::PatternTrackParameters  & Tb,
 Trk::PropDirection             D ,
 const MagneticFieldProperties& M , 
 ParticleHypothesis               ) const 
{
  double S;
  return propagateRungeKutta(true,Ta,Su,Tb,D,M,S);
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for simple track parameters and covariance matrix propagation
// Ta->Su = Tb with step to surface calculation
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::propagate
(Trk::PatternTrackParameters  & Ta,
 const Trk::Surface           & Su,
 Trk::PatternTrackParameters  & Tb,
 Trk::PropDirection             D ,
 const MagneticFieldProperties& M , 
 double                       & S , 
 ParticleHypothesis               ) const 
{
  return propagateRungeKutta(true,Ta,Su,Tb,D,M,S);
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for simple track parameters propagation without covariance matrix
// Ta->Su = Tb
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::propagateParameters
(Trk::PatternTrackParameters  & Ta,
 const Trk::Surface           & Su,
 Trk::PatternTrackParameters  & Tb, 
 Trk::PropDirection             D ,
 const MagneticFieldProperties& M ,
 ParticleHypothesis               ) const 
{
  double S;
  return propagateRungeKutta(false,Ta,Su,Tb,D,M,S);
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for simple track parameters propagation without covariance matrix
// Ta->Su = Tb with step calculation
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::propagateParameters
(Trk::PatternTrackParameters  & Ta,
 const Trk::Surface           & Su,
 Trk::PatternTrackParameters  & Tb, 
 Trk::PropDirection             D ,
 const MagneticFieldProperties& M ,
 double                       & S , 
 ParticleHypothesis               ) const 
{
  return propagateRungeKutta(false,Ta,Su,Tb,D,M,S);
}

/////////////////////////////////////////////////////////////////////////////////
// Global positions calculation inside CylinderBounds
// where mS - max step allowed if mS > 0 propagate along    momentum
//                                mS < 0 propogate opposite momentum
/////////////////////////////////////////////////////////////////////////////////

void Trk::RungeKuttaPropagator::globalPositions 
(std::list<Trk::GlobalPosition>   & GP,
 const Trk::PatternTrackParameters& Tp,
 const MagneticFieldProperties    & M,
 const CylinderBounds             & CB,
 double                             mS,
 ParticleHypothesis                   ) const
{
  Trk::RungeKuttaUtils utils;
  double P[45]; if(!utils.transformLocalToGlobal(false,Tp,P)) return;

  m_direction = fabs(mS);
  if(mS > 0.) globalOneSidePositions(GP,P,M,CB, mS);
  else        globalTwoSidePositions(GP,P,M,CB,-mS);
}

/////////////////////////////////////////////////////////////////////////////////
// GlobalPostions and steps for set surfaces
/////////////////////////////////////////////////////////////////////////////////

void Trk::RungeKuttaPropagator::globalPositions
(const PatternTrackParameters                 & Tp,
 std::list<const Trk::Surface*>               & SU,
 std::list< std::pair<GlobalPosition,double> >& GP,
 const Trk::MagneticFieldProperties           & M ,
 ParticleHypothesis                               ) const
{
  m_magneticFieldProperties = &M   ;
  m_magneticFieldTool       = 0    ;
  m_direction               = 0.   ;
  m_mcondition              = false;
  if(m_magneticFieldProperties && m_magneticFieldProperties->magneticFieldMode()!=Trk::NoField) m_mcondition = true;

  Trk::RungeKuttaUtils utils;

  double Step = 0.,P[45]; if(!utils.transformLocalToGlobal(false,Tp,P)) return;

  const Trk::DiscSurface*         dis;
  const Trk::PlaneSurface*        pla;
  const Trk::PerigeeSurface*      per;
  const Trk::CylinderSurface*     cyl;
  const Trk::StraightLineSurface* lin;

  std::list<const Trk::Surface*>::iterator su = SU.begin(), sue = SU.end();

  // Loop trough all input surfaces
  //
  for(; su!=sue; ++su) {
    
    if((pla=dynamic_cast<const Trk::PlaneSurface*>            (*su))) {

      const Trk::GlobalPosition&  R = pla->center(); 
      const GlobalDirection& A = pla->normal();
      double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
      double s[4];
      if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
      else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
      if(!propagateWithJacobian(false,1,s,P,Step))  return;
    }
    else if((lin=dynamic_cast<const Trk::StraightLineSurface*>(*su))) {

      const Trk::GlobalPosition&      R = lin->center();
      const Trk::Transform3D&  T = lin->transform(); 
      double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
      if(!propagateWithJacobian(false,0,s,P,Step))  return;
    }
    else if((dis=dynamic_cast<const Trk::DiscSurface*>        (*su))) {

      const Trk::GlobalPosition&  R = dis->center  (); 
      const GlobalDirection& A = dis->normal  ();
      double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
      double s[4];
      if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
      else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
      if(!propagateWithJacobian(false,1,s,P,Step))  return;
    }
    else if((cyl=dynamic_cast<const Trk::CylinderSurface*>    (*su))) {

      const Trk::Transform3D&  T = cyl->transform(); 
      double r0[3] = {P[0],P[1],P[2]};
      double s [9] = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cyl->bounds().r(),1.,0.};
      if(!propagateWithJacobian(false,2,s,P,Step))  return;

      // For cylinder we do test for next cross point
      //
      if(cyl->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cyl,r0,P)) {
	s[8] = 0.; if(!propagateWithJacobian(false,2,s,P,Step)) return;
      }
    }

    else if((per=dynamic_cast<const Trk::PerigeeSurface*>     (*su))) {

      const Trk::GlobalPosition& R = per->center();  
      double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
      if(!propagateWithJacobian(false,0,s,P,Step))  return;
    }
    else return;

    Trk::GlobalPosition gp(P[0],P[1],P[2]); GP.push_back(std::make_pair(gp,Step));
  }
}

/////////////////////////////////////////////////////////////////////////////////
// Main function for simple track propagation with or without jacobian
// Ta->Su = Tb for pattern track parameters
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::propagateRungeKutta
(bool                           useJac,
 Trk::PatternTrackParameters  & Ta    ,
 const Trk::Surface           & Su    ,
 Trk::PatternTrackParameters  & Tb    ,
 Trk::PropDirection             D     ,
 const MagneticFieldProperties& M     ,
 double                       & Step  ) const 
{  
  if(&Su == Ta.associatedSurface()) {Tb = Ta; return true;}

  if(useJac && !Ta.iscovariance()) useJac = false;

  m_magneticFieldProperties = &M;
  m_magneticFieldTool       = 0 ;
  m_direction               = D ; 

  if( m_usegradient && useJac) {
    m_magneticFieldTool = dynamic_cast<const Trk::IMagneticFieldTool*>(M.magneticFieldTool());
  }
  
  m_mcondition = false;
  if(m_magneticFieldProperties && m_magneticFieldProperties->magneticFieldMode()!=Trk::NoField) m_mcondition = true;

  Trk::RungeKuttaUtils utils;

  double P[45]; if(!utils.transformLocalToGlobal(useJac,Ta,P)) return false;
  Step = 0.;
  const Trk::DiscSurface*         dis = 0;
  const Trk::PlaneSurface*        pla = 0;
  const Trk::PerigeeSurface*      per = 0;
  const Trk::CylinderSurface*     cyl = 0;
  const Trk::StraightLineSurface* lin = 0;

  if      ((pla=dynamic_cast<const Trk::PlaneSurface*>       (&Su))) {

    const Trk::GlobalPosition&  R = pla->center(); 
    const GlobalDirection& A = pla->normal();
    double     d  = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(useJac,1,s,P,Step)) return false;
  }
  else if ((lin=dynamic_cast<const Trk::StraightLineSurface*>(&Su))) {

    const Trk::GlobalPosition&      R = lin->center();
    const Trk::Transform3D&  T = lin->transform(); 
    double  s[6]  = {R.x(),R.y(),R.z(),T.xz(),T.yz(),T.zz()};
    if(!propagateWithJacobian(useJac,0,s,P,Step)) return false;
  }
  else if ((dis=dynamic_cast<const Trk::DiscSurface*>        (&Su))) {

    const Trk::GlobalPosition&  R = dis->center  (); 
    const GlobalDirection& A = dis->normal  ();
    double      d = R.x()*A.x()+R.y()*A.y()+R.z()*A.z();
    double s[4];
    if(d>=0.) {s[0]= A.x(); s[1]= A.y(); s[2]= A.z(); s[3]= d;}
    else      {s[0]=-A.x(); s[1]=-A.y(); s[2]=-A.z(); s[3]=-d;}
    if(!propagateWithJacobian(useJac,1,s,P,Step)) return false;
  }
  else if ((cyl=dynamic_cast<const Trk::CylinderSurface*>    (&Su))) {

    const Trk::Transform3D&  T = cyl->transform(); 
    double r0[3] = {P[0],P[1],P[2]};
    double s[9]  = {T.dx(),T.dy(),T.dz(),T.xz(),T.yz(),T.zz(),cyl->bounds().r(),D,0.};
    if(!propagateWithJacobian(useJac,2,s,P,Step)) return false;

    // For cylinder we do test for next cross point
    //
    if(cyl->bounds().halfPhiSector() < 3.1 && newCrossPoint(*cyl,r0,P)) {
      s[8] = 0.; if(!propagateWithJacobian(useJac,2,s,P,Step)) return false;
    }
  }
  else if ((per=dynamic_cast<const Trk::PerigeeSurface*>     (&Su))) {

    const Trk::GlobalPosition& R = per->center();  
    double  s[6]  = {R.x(),R.y(),R.z(),0.,0.,1.};
    if(!propagateWithJacobian(useJac,0,s,P,Step)) return false;
  }
  else return false;

  if(m_direction && (m_direction*Step)<0.) return false;

  // Common transformation for all surfaces (angles and momentum)
  //
  if(useJac) {
    double p=1./P[6]; P[35]*=p; P[36]*=p; P[37]*=p; P[38]*=p; P[39]*=p; P[40]*=p;
  }
  double p[5],Jac[21]; utils.transformGlobalToLocal(useJac,P,p);

  // Surface dependent transformations (local coordinates)
  //
  if     (pla) utils.transformGlobalToLocal(useJac,P,*pla,p,Jac);
  else if(lin) utils.transformGlobalToLocal(useJac,P,*lin,p,Jac); 
  else if(dis) utils.transformGlobalToLocal(useJac,P,*dis,p,Jac);
  else if(cyl) utils.transformGlobalToLocal(useJac,P,*cyl,p,Jac);
  else         utils.transformGlobalToLocal(useJac,P,*per,p,Jac);

  // New simple track parameters production
  //
  Tb.setParameters(&Su,p); 
  if(useJac) {
    Tb.newCovarianceMatrix(Ta,Jac);
    const double* cv = Tb.cov();
    if( cv[0]<=0. || cv[2]<=0. || cv[5]<=0. || cv[9]<=0. || cv[14]<=0.) return false;
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////
// Test new cross point
/////////////////////////////////////////////////////////////////////////////////

bool Trk::RungeKuttaPropagator::newCrossPoint
(const Trk::CylinderSurface& Su,
 const double              * Ro,
 const double              * P ) const
{
  const double pi = 3.1415927, pi2=2.*pi; 
  const Trk::Transform3D&  T = Su.transform();
  double Ax[3] = {T.xx(),T.yx(),T.zx()};
  double Ay[3] = {T.xy(),T.yy(),T.zy()};

  double R     = Su.bounds().r();
  double x     = Ro[0]-T.dx();
  double y     = Ro[1]-T.dy();
  double z     = Ro[2]-T.dz();

  double RC    = x*Ax[0]+y*Ax[1]+z*Ax[2];
  double RS    = x*Ay[0]+y*Ay[1]+z*Ay[2];

  if( (RC*RC+RS*RS) <= (R*R) ) return false;
  
  x           = P[0]-T.dx();
  y           = P[1]-T.dy();
  z           = P[2]-T.dz();
  RC          = x*Ax[0]+y*Ax[1]+z*Ax[2];
  RS          = x*Ay[0]+y*Ay[1]+z*Ay[2];
  double dF   = fabs(atan2(RS,RC)-Su.bounds().averagePhi());
  if(dF > pi) dF = pi2-pi;
  if(dF <= Su.bounds().halfPhiSector()) return false;
  return true;
}

/////////////////////////////////////////////////////////////////////////////////
// Build new track parameters without propagation
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::buildTrackParametersWithoutPropagation
(const Trk::TrackParameters        & Tp,
 const Trk::MeasuredTrackParameters* Mp) const
{
  const Trk::AtaPlane*        pla = 0;

  if((pla = dynamic_cast<const Trk::AtaPlane*>        (&Tp))) {
    if(!Mp) return new Trk::AtaPlane       (*pla);
    return new Trk::MeasuredAtaPlane       (*dynamic_cast<const Trk::MeasuredAtaPlane*>       (Mp)); 
  }

  const Trk::AtaStraightLine* lin = 0;

  if((lin = dynamic_cast<const Trk::AtaStraightLine*> (&Tp))) {
    if(!Mp) return new Trk::AtaStraightLine(*lin);
    return new Trk::MeasuredAtaStraightLine(*dynamic_cast<const Trk::MeasuredAtaStraightLine*>(Mp));
  }

  const Trk::AtaDisc*         dis = 0;

  if((dis = dynamic_cast<const Trk::AtaDisc*>         (&Tp))) {
    if(!Mp) return new Trk::AtaDisc        (*dis);
    return new Trk::MeasuredAtaDisc        (*dynamic_cast<const Trk::MeasuredAtaDisc*>        (Mp));
  }

  const Trk::AtaCylinder*     cyl = 0;

  if((cyl = dynamic_cast<const Trk::AtaCylinder*>     (&Tp))) {
    if(!Mp) return new Trk::AtaCylinder    (*cyl); 
    return new Trk::MeasuredAtaCylinder    (*dynamic_cast<const Trk::MeasuredAtaCylinder*>    (Mp));
  }

  const Trk::Perigee*         per = 0;

  if((per = dynamic_cast<const Trk::Perigee*>         (&Tp))) {
    if(!Mp) return new Trk::Perigee        (*per);
    return new Trk::MeasuredPerigee        (*dynamic_cast<const Trk::MeasuredPerigee*>        (Mp));
  }
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////
// Build new neutral track parameters without propagation
/////////////////////////////////////////////////////////////////////////////////

const Trk::NeutralParameters* Trk::RungeKuttaPropagator::buildTrackParametersWithoutPropagation
(const Trk::NeutralParameters        & Tp,
 const Trk::MeasuredNeutralParameters* Mp) const
{
  
  const Trk::NeutralAtaPlane*        pla = 0;

  if((pla = dynamic_cast<const Trk::NeutralAtaPlane*>           (&Tp))) {
    if(!Mp) return new Trk::NeutralAtaPlane       (*pla);
    return new Trk::MeasuredNeutralAtaPlane       
      (*dynamic_cast<const Trk::MeasuredNeutralAtaPlane*>       (Mp)); 
  }
  
  const Trk::NeutralAtaStraightLine* lin = 0;

  if((lin = dynamic_cast<const Trk::NeutralAtaStraightLine*>    (&Tp))) {
    if(!Mp) return new Trk::NeutralAtaStraightLine(*lin);
    return new Trk::MeasuredNeutralAtaStraightLine
      (*dynamic_cast<const Trk::MeasuredNeutralAtaStraightLine*>(Mp));
  }

  const Trk::NeutralAtaDisc*         dis = 0;

  if((dis = dynamic_cast<const Trk::NeutralAtaDisc*>            (&Tp))) {
    if(!Mp) return new Trk::NeutralAtaDisc        (*dis);
    return new Trk::MeasuredNeutralAtaDisc        
      (*dynamic_cast<const Trk::MeasuredNeutralAtaDisc*>        (Mp));
  }

  
  const Trk::NeutralAtaCylinder*     cyl = 0;
  
  if((cyl = dynamic_cast<const Trk::NeutralAtaCylinder*>        (&Tp))) {
    if(!Mp) return new Trk::NeutralAtaCylinder    (*cyl); 
    return new Trk::MeasuredNeutralAtaCylinder     
     (*dynamic_cast<const Trk::MeasuredNeutralAtaCylinder*>    (Mp));
  }
  
  const Trk::NeutralPerigee*         per = 0;

  if((per = dynamic_cast<const Trk::NeutralPerigee*>            (&Tp))) {
    if(!Mp) return new Trk::NeutralPerigee        (*per);
    return new Trk::MeasuredNeutralPerigee        
      (*dynamic_cast<const Trk::MeasuredNeutralPerigee*>        (Mp));
  }
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////
// Step estimator take into accout curvature
/////////////////////////////////////////////////////////////////////////////////

double Trk::RungeKuttaPropagator::stepEstimatorWithCurvature
(int kind,double* Su,const double* P,bool& Q) const
{
  // Straight step estimation
  //
  Trk::RungeKuttaUtils utils;
  double  Step = utils.stepEstimator(kind,Su,P,Q); if(!Q) return 0.; 
  double AStep = fabs(Step);
  if( kind || AStep < m_straightStep || !m_mcondition ) return Step;

  const double* SA = &P[42]; // Start direction
  double S = .5*Step;
  
  double Ax    = P[3]+S*SA[0];
  double Ay    = P[4]+S*SA[1];
  double Az    = P[5]+S*SA[2];
  double As    = 1./sqrt(Ax*Ax+Ay*Ay+Az*Az);
  double PN[6] = {P[0],P[1],P[2],Ax*As,Ay*As,Az*As};
  double StepN = utils.stepEstimator(kind,Su,PN,Q); if(!Q) {Q = true; return Step;}
  if(fabs(StepN) < AStep) return StepN; return Step;
} 

/////////////////////////////////////////////////////////////////////////////////
// Global positions calculation inside CylinderBounds (one side)
// where mS - max step allowed if mS > 0 propagate along    momentum
//                                mS < 0 propogate opposite momentum
/////////////////////////////////////////////////////////////////////////////////

void Trk::RungeKuttaPropagator::globalOneSidePositions 
(std::list<Trk::GlobalPosition>& GP,
 const double                  * P,
 const MagneticFieldProperties & M ,
 const CylinderBounds          & CB,
 double                          mS,
 ParticleHypothesis                ) const
{
  m_magneticFieldProperties = &M;

  double Pm[45]; for(int i=0; i!=7; ++i) Pm[i]=P[i];

  double       W     = 0.                 ; // way
  double       R2max = CB.r()*CB.r()      ; // max. radius**2 of region
  double       Zmax  = CB.halflengthZ()   ; // max. Z         of region
  double       R2    = P[0]*P[0]+P[1]*P[1]; // Start radius**2
  double       Dir   = P[0]*P[3]+P[1]*P[4]; // Direction
  double       S     = mS                 ; // max step allowed
  double       R2m   = R2                 ;

  if(M.magneticFieldMode()!=Trk::NoField && fabs(P[6]) > .1) return; 

  // Test position of the track  
  //
  if(fabs(P[2]) > Zmax || R2 > R2max) return;

  Trk::GlobalPosition g0(P[0],P[1],P[2]); GP.push_back(g0);

  bool   per = false; if(fabs(Dir)<.00001) per = true;
  bool   InS = false;

  int niter = 0;
  int sm    = 1;
  for(int s = 0; s!=2; ++s) {

    if(s) {if(per) break; S = -S;}
    double p[45]; for(int i=0; i!=7; ++i) p[i]=P[i]; p[42]=p[43]=p[44]=0.;
    
    while(W<100000. && ++niter < 1000) {
      
      if(M.magneticFieldMode()!=Trk::NoField) {
	W+=(S=rungeKuttaStep  (0,S,p,InS)); 
      }
      else {
	W+=(S=straightLineStep(0,    S,p)); 
      }       
      if(InS && fabs(2.*S)<mS) S*=2.;

      Trk::GlobalPosition g(p[0],p[1],p[2]); 
      if(!s) GP.push_back(g); else GP.push_front(g);
   
      // Test position of the track  
      //
      R2 = p[0]*p[0]+p[1]*p[1];

      if(R2 < R2m) {
	Pm[0]=p[0]; Pm[1]=p[1]; Pm[2]=p[2]; 
	Pm[3]=p[3]; Pm[4]=p[4]; Pm[5]=p[5]; R2m = R2; sm = s;
      }
      if(fabs(p[2]) > Zmax || R2 > R2max) break;
      if(!s && P[3]*p[3]+P[4]*p[4] < 0. ) break;

      // Test perigee 
      //
      if((p[0]*p[3]+p[1]*p[4])*Dir < 0.) {
	if(s) break; per = true;
      }
    }
  }

  if(R2m < 400.) return;

  // Propagate to perigee
  //
  Pm[42]=Pm[43]=Pm[44]=0.;
  per = false;

  for(int s = 0; s!=3; ++s) {

    double A = (1.-Pm[5])*(1.+Pm[5]); if(A==0.) break;
    S        = -(Pm[0]*Pm[3]+Pm[1]*Pm[4])/A;
    if(fabs(S) < 1. || ++niter > 1000) break;

    if(M.magneticFieldMode()!=Trk::NoField) {
      W+=(S=rungeKuttaStep  (0,S,Pm,InS)); 
    }
    else {
      W+=(S=straightLineStep(0,S,Pm)); 
    }       
    per = true;
  }

  if(per) {
    if(sm) {Trk::GlobalPosition gf(Pm[0],Pm[1],Pm[2]); GP.front() = gf;}
    else   {Trk::GlobalPosition gf(Pm[0],Pm[1],Pm[2]); GP.back () = gf;} 
  }
  else   {
    double x = GP.front().x() , y = GP.front().y();
    if( (x*x+y*y) > (Pm[0]*Pm[0]+Pm[1]*Pm[1]) ) {
     if(sm) GP.pop_front();
     else   GP.pop_back ();
    }
  }
  return;
}

/////////////////////////////////////////////////////////////////////////////////
// Global positions calculation inside CylinderBounds (one side)
// where mS - max step allowed if mS > 0 propagate along    momentum
//                                mS < 0 propogate opposite momentum
/////////////////////////////////////////////////////////////////////////////////

void Trk::RungeKuttaPropagator::globalTwoSidePositions 
(std::list<Trk::GlobalPosition>& GP,
 const double                  * P,
 const MagneticFieldProperties & M ,
 const CylinderBounds          & CB,
 double                          mS,
 ParticleHypothesis                ) const
{
  m_magneticFieldProperties = &M;

  double       W     = 0.                 ; // way
  double       R2max = CB.r()*CB.r()      ; // max. radius**2 of region
  double       Zmax  = CB.halflengthZ()   ; // max. Z         of region
  double       R2    = P[0]*P[0]+P[1]*P[1]; // Start radius**2
  double       S     = mS                 ; // max step allowed

  if(M.magneticFieldMode()!=Trk::NoField && fabs(P[6]) > .1) return; 

  // Test position of the track  
  //
  if(fabs(P[2]) > Zmax || R2 > R2max) return;

  Trk::GlobalPosition g0(P[0],P[1],P[2]); GP.push_back(g0);

  bool   InS = false;

  int niter = 0;
  for(int s = 0; s!=2; ++s) {

    if(s) {S = -S;}
    double p[45]; for(int i=0; i!=7; ++i) p[i]=P[i]; p[42]=p[43]=p[44]=0.;
    
    while(W<100000. && ++niter < 1000) {
      
      if(M.magneticFieldMode()!=Trk::NoField) {
	W+=(S=rungeKuttaStep  (0,S,p,InS)); 
      }
      else {
	W+=(S=straightLineStep(0,    S,p)); 
      }       
      if(InS && fabs(2.*S)<mS) S*=2.;

      Trk::GlobalPosition g(p[0],p[1],p[2]); 
      if(!s) GP.push_back(g); else GP.push_front(g);
   
      // Test position of the track  
      //
      R2 = p[0]*p[0]+p[1]*p[1];
      if(fabs(p[2]) > Zmax || R2 > R2max) break;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////
// Track parameters in cross point preparation
/////////////////////////////////////////////////////////////////////////////////

const Trk::TrackParameters* Trk::RungeKuttaPropagator::crossPoint
(const TrackParameters    & Tp,
 std::vector<DestSurf>    & SU,
 std::vector<unsigned int>& So,
 double                   * P ,
 std::pair<double,int>    & SN) const
{
  double*   R = &P[ 0]   ;  // Start coordinates
  double*   A = &P[ 3]   ;  // Start directions
  double*  SA = &P[42]   ;  // d(directions)/dStep
  double Step = SN.first ; 
  int      N  = SN.second; 

  double As[3],Rs[3];
  
  As[0] = A[0]+SA[0]*Step; 
  As[1] = A[1]+SA[1]*Step;
  As[2] = A[2]+SA[2]*Step;
  
  double CBA  = 1./sqrt(As[0]*As[0]+As[1]*As[1]+As[2]*As[2]);
  
  Rs[0] = R[0]+Step*(As[0]-.5*Step*SA[0]); As[0]*=CBA;
  Rs[1] = R[1]+Step*(As[1]-.5*Step*SA[1]); As[1]*=CBA;
  Rs[2] = R[2]+Step*(As[2]-.5*Step*SA[2]); As[2]*=CBA;

  Trk::GlobalPosition   pos(Rs[0],Rs[1],Rs[2]);
  Trk::GlobalDirection  dir(As[0],As[1],As[2]);

  Trk::DistanceSolution ds = SU[N].first->straightLineDistanceEstimate(pos,dir,SU[N].second);  
  if(ds.currentDistance(false) > .010) return 0;

  P[0] = Rs[0]; A[0] = As[0];
  P[1] = Rs[1]; A[1] = As[1];
  P[2] = Rs[2]; A[2] = As[2];

  So.push_back(N);

  // Transformation track parameters
  //
  Trk::RungeKuttaUtils utils;
  
  const Trk::MeasuredTrackParameters* 
    Mp = dynamic_cast<const Trk::MeasuredTrackParameters*>(&Tp);

  bool useJac; Mp ? useJac = true : useJac = false;

  if(useJac) {
    double d=1./P[6]; P[35]*=d; P[36]*=d; P[37]*=d; P[38]*=d; P[39]*=d; P[40]*=d;
  }
  double p[5],Jac[25]; 
  utils.transformGlobalToLocal(useJac,P,p);

  // Surface dependent transformations (local coordinates)
  //
  const Trk::DiscSurface*         dis = 0;
  const Trk::PlaneSurface*        pla = 0;
  const Trk::PerigeeSurface*      per = 0;
  const Trk::CylinderSurface*     cyl = 0;
  const Trk::StraightLineSurface* lin = 0;

  if      ((pla=dynamic_cast<const Trk::PlaneSurface*>       (SU[N].first))) {
    utils.transformGlobalToLocal(useJac,P,*pla,p,Jac);
  }
  else if ((lin=dynamic_cast<const Trk::StraightLineSurface*>(SU[N].first))) {
    utils.transformGlobalToLocal(useJac,P,*lin,p,Jac); 
  }
  else if ((dis=dynamic_cast<const Trk::DiscSurface*>        (SU[N].first))) {
    utils.transformGlobalToLocal(useJac,P,*dis,p,Jac);
  }
  else if ((cyl=dynamic_cast<const Trk::CylinderSurface*>    (SU[N].first))) {
    utils.transformGlobalToLocal(useJac,P,*cyl,p,Jac); 
  }
  else if ((per=dynamic_cast<const Trk::PerigeeSurface*>     (SU[N].first))) {
    utils.transformGlobalToLocal(useJac,P,*per,p,Jac);
  }
  else return 0;

  if(!useJac) {

    if(pla) return new Trk::AtaPlane       (p[0],p[1],p[2],p[3],p[4],*pla);
    if(lin) return new Trk::AtaStraightLine(p[0],p[1],p[2],p[3],p[4],*lin); 
    if(dis) return new Trk::AtaDisc        (p[0],p[1],p[2],p[3],p[4],*dis); 
    if(cyl) return new Trk::AtaCylinder    (p[0],p[1],p[2],p[3],p[4],*cyl); 
            return new Trk::Perigee        (p[0],p[1],p[2],p[3],p[4],*per); 
  }

  if(!&Mp->localErrorMatrix() || !&Mp->localErrorMatrix().covariance() || Mp->localErrorMatrix().covariance().num_row()!=5) return 0; 

  Trk::CovarianceMatrix* C = utils.newCovarianceMatrix(Jac,Mp->localErrorMatrix().covariance());
  if(C->fast(1,1)<=0. || C->fast(2,2)<=0. || C->fast(3,3)<=0. || C->fast(4,4)<=0. || C->fast(5,5)<=0.) {
    delete C; return 0;
  }
  Trk::ErrorMatrix* e = new Trk::ErrorMatrix(C);

  if(pla) return new Trk::MeasuredAtaPlane       (p[0],p[1],p[2],p[3],p[4],*pla,e);
  if(lin) return new Trk::MeasuredAtaStraightLine(p[0],p[1],p[2],p[3],p[4],*lin,e); 
  if(dis) return new Trk::MeasuredAtaDisc        (p[0],p[1],p[2],p[3],p[4],*dis,e); 
  if(cyl) return new Trk::MeasuredAtaCylinder    (p[0],p[1],p[2],p[3],p[4],*cyl,e); 
          return new Trk::MeasuredPerigee        (p[0],p[1],p[2],p[3],p[4],*per,e);
}

/////////////////////////////////////////////////////////////////////////////////
// Step reduction from STEP propagator
// Input information ERRx,ERRy,ERRz,current step   
/////////////////////////////////////////////////////////////////////////////////

double Trk::RungeKuttaPropagator::stepReduction(const double* E) const
{
  double dlt2  = m_dlt*m_dlt;
  double dR2   = (E[0]*E[0]+E[1]*E[1]+E[2]*E[2])*E[3]*E[3]*4.; if(dR2 < 1.e-40) dR2 = 1.e-40;
  double cS    = std::pow(dlt2/dR2,0.125);
  if(cS < .25) cS = .25; 
  if((dR2 > 16.*dlt2 && cS < 1.) || cS >=3.) return cS;
  return 1.;
}
