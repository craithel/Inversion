#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "../nrutil.h"
#include "header_useful.h"

#define nparam 6							//6 parameter points = 5 polytropic segemnets
#define nEpsilon 96							//Number of sets of central densities to cycle thru
#define nODEs 6					 			//Number of coupled ODEs to solve	
#define nstep 100000							//Max number of steps to take
#define lines 1000							//Max number of expected lines in EoS file 
#define hscale 1.0e-3							//Scaling for step size, h 
#define Msolar 1.988435e33						//Mass of sun in grams		
#define clight 2.99792458e10						//Speed of light in cm/s
#define Ggrav 6.6738480e-8						//Gravitational Constant (CGS)
#define MoverR Msolar*Ggrav/(1.0e5*clight*clight)			//Natural units for size of neutron star: (1 solar mass * G)/(1km * c^2) 
#define rho_char 4.747039e+17						//Characteristic mass density for neutron star in g/cm^3: (1 solar mass)/(4pi/3* 1km^3)
#define eps_char 4.747039e+17						//Characteristic energy density = mass density * c^2
#define p_char (eps_char*clight*clight)					//Characteristic pressure for neutron star in g s^2/cm
#define eps_min 3.0e-4							//Lowest rho_c to evaluate
#define r_min 1.0e-4							//Starting radius
#define ACCURACY 1.0e-15
#define rho_ns 5.688e-4							//Unitless nuclear saturation density = (rho_ns / solar mass)*(4/3 pi (1km)^3)

extern double (*Ppts);								//Initialize P0 and the 3 free params for the EOS
extern double (*rhopts);
extern double (*gamma0_param);
extern double (*acoef_param);
extern double (*p_SLY);
extern double (*epsilon_SLY);
extern double (*rho_SLY);
extern double (*y_m);						   		//Initialize global array to save solutions for mass
extern double (*inertia);						   	//Initialize global array to save solutions for TRIAL mom of inertia
extern double (*rr);    					   		//Initialize global array to save steps taken
extern double p_ns;
extern int numlinesSLY;							


extern void tov();
extern double findEps0_inSLY(double rho0);					//Prototype func to find Epsilon corresponding to a Rho using SLy
extern double param_massToEnergy(double massdensity);				//Prototype func for Eps(rho), assuming a polytropic EoS
