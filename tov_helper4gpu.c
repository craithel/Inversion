/* C. Raithel, May 2016
   
   This program does all the one-time computations formerly performed in 
   tov_polyparam.c. By separating these functions into this program,
   tov_polyparam.c will run more quickly when called repeatedly (e.g., in 
   the MCMC chain of inversion.c)

*/

#include "tov_dev.h"

extern void readinSLY(double *p_SLY_local, double *eps_SLY_local, double *rho_SLY_local, int *nSLY_local)
/* Read in tabulated data for the low-density regime, using the EoS Sly. */
{

	int i=1,j;
	FILE *file;
	file = fopen("/gsfs1/xdisk/craithel/sly.dat","rt");
	while ( fscanf(file,"%le %le %le", &rho_SLY_local[i], &p_SLY_local[i], &eps_SLY_local[i]) == 3 ) i++;	//Column order: mass density, P, energy density
	*nSLY_local = i-1;							//set global variable to save number of lines in this file
	for (j=1; j<=i-1; j++)
	{
		p_SLY_local[j] = p_SLY_local[j]*clight*clight/p_char;				//convert to dim'less units (note P=P/c^2 in .dat files)
		eps_SLY_local[j] /= eps_char;						//convert to dim'less units
		rho_SLY_local[j] /= rho_char;
	}
	fclose(file);								//Close EoS file

}

extern void getRhoPts(double *rhopts_local)
/*
  Compute the fiducial densities: evenly spaced in the log of density
  between rho_sat and 7.4*rho_sat
*/
{
	int j;
	double rho0, rho3;

	rho0=1.0*rho_ns;	
	rho3=7.4*rho_ns;

	rhopts_local[1] = rho0;
	for (j=2; j<=nparam; j++) rhopts_local[j] = rhopts_local[1]*pow(10.0,log10(rho3/rhopts_local[1])* (j-1)/(nparam-1));
}
