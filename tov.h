/* Header file for the programs: tov_polyparam.c and tov_helper.c */


#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "../nrutil.h"
#include "header_useful.h"

#define nparam 6							//6 parameter points = 5 polytropic segemnets
#define nEpsilon 30							//Number of sets of central densities to cycle thru
#define nODEs 2					 			//Number of coupled ODEs to solve	
#define nstep 100000							//Max number of steps to take
#define lines 105							//Max number of expected lines in EoS file 
#define hscale 1.0e-3							//Scaling for step size, h 
#define Msolar 1.988435e33						//Mass of sun in grams		
#define clight 2.99792458e10						//Speed of light in cm/s
#define Ggrav 6.6738480e-8						//Gravitational Constant (CGS)
#define MoverR Msolar*Ggrav/(1.0e5*clight*clight)			//Natural units for size of neutron star: (1 solar mass * G)/(1km * c^2) 
#define rho_char 4.747039e+17						//Characteristic mass density for neutron star in g/cm^3: (1 solar mass)/(4pi/3* 1km^3)
#define eps_char 4.747039e+17						//Characteristic energy density = mass density * c^2
#define p_char (eps_char*clight*clight)					//Characteristic pressure for neutron star in g s^2/cm
#define rho_ns 5.688e-4							//Unitless nuclear saturation density = (rho_ns / solar mass)*(4/3 pi (1km)^3)
#define eps_min rho_ns//7.6e-4							//Lowest rho_c to evaluate
#define r_min 1.0e-4							//Starting radius
#define ACCURACY 1.0e-15


extern double *rhopts; 
extern double *p_SLY, *epsilon_SLY, *rho_SLY;
extern double *initRho, *initM, *initEpsilon, *centralP;
extern double Pedge, r_start, p_ns;
extern int numlinesSLY;	

extern void tov(double Ppts_in[], double *rr_in, double *y_m_in, double *gamma0_param_in, double *acoef_param_in);
extern double param_massToEnergy(double massdensity, double Ppts[], double gamma0_param[], double acoef_param[]);				//Prototype func for Eps(rho), assuming a polytropic EoS
extern double EOSpressure(double massdensity, double Ppts[], double gamma0_param[], int useparam);
extern void getGammaA(double Ppts[], double *gamma0_param, double *acoef_param);	//Prototype func to calculate gamma and a_coef along each segment given Ppts
extern void getRhoPts(double *rhopts);
extern void readinSLY(double *p_SLY, double *epsilon_SLY, double *rho_SLY, int *numlinesSLY);
extern double findEps0_inSLY(double rho0);					//Prototype func to find Epsilon corresponding to a Rho using SLy
extern void edgeConditions(double *r_start, double *Pedge);
extern void initConditions(double *initM, double *initRho, double *initEpsilon, double *centralP, double *Ppts, double *gamma0_param, double *acoef_param);
