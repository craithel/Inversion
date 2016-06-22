/* Header file for the programs: tov_polyparam_dev.cu  */

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
#define eps_min 7.6e-4							//Lowest rho_c to evaluate
#define r_min 1.0e-4							//Starting radius
#define ACCURACY 1.0e-15
#define rho_ns 5.688e-4							//Unitless nuclear saturation density = (rho_ns / solar mass)*(4/3 pi (1km)^3)
#define r_start 1.47654e-4			//r_start=r_min*Ggrav*Msolar/clight/clight/1.0e5
#define Pedge 1.30812e-15			//ACCURACY*pow(clight,8.0)/(Ggrav*Ggrav*Ggrav*Msolar*Msolar) / pchar

#ifdef __cplusplus
extern "C" {
#endif
extern void getRhoPts(double *rhopts_local);
extern void readinSLY(double *p_SLY_local, double *eps_SLY_local, double *rho_SLY_local, int *nSLY_local);
#ifdef __cplusplus
}
#endif


