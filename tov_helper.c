/* C. Raithel, May 2016
   
   This program does all the one-time computations formerly performed in 
   tov_polyparam.c. By separating these functions into this program,
   tov_polyparam.c will run more quickly when called repeatedly (e.g., in 
   the MCMC chain of inversion.c)

*/

#include "header.h"

extern void initConditions(double *initM, double *initRho, double *initEpsilon, double *centralP, double *Ppts, double *gamma0_param, double *acoef_param)
{

	int i,j;
	getGammaA(Ppts, gamma0_param, acoef_param);	       //Get gamma and 'a' coefs for this set of Ppts
	initRho[1] = 1.0*eps_min;
	for (i=2;i<=nEpsilon;i++) initRho[i] = initRho[i-1]*1.07;	//Scale starting mass density values
	for (j=1;j<=nEpsilon;j++)					//Convert init rho to eps
	{	
		if (initRho[j] <= rhopts[1])
		{
			initEpsilon[j] = findEps0_inSLY(initRho[j]);
			centralP[j] = EOSpressure(initRho[j],Ppts, gamma0_param, 0);
		}
		else	
		{		
			initEpsilon[j] = param_massToEnergy(initRho[j],Ppts,gamma0_param,acoef_param);
			centralP[j] = EOSpressure(initRho[j],Ppts,gamma0_param, 1);
		}
		initM[j] = r_start*r_start*r_start*initEpsilon[j];		//Initial masses enclosed by r_start
	}

}


extern void edgeConditions(double *r_start, double *Pedge)
{
	*Pedge=ACCURACY*pow(clight,8.0)/(Ggrav*Ggrav*Ggrav*Msolar*Msolar);	//P_edge criteria in cgs 
	*Pedge/=p_char;								//P_edge criteria dimless
	*r_start=r_min*Ggrav*Msolar/clight/clight/1.0e5;				
}

extern void readinSLY(double *p_SLY, double *epsilon_SLY, double *rho_SLY, int *numlinesSLY)
/* Read in tabulated data for the low-density regime, using the EoS Sly. */
{

	int i=1,j;
	FILE *file;
	file = fopen("/gsfs1/xdisk/craithel/sly.dat","rt");
	while ( fscanf(file,"%le %le %le", &rho_SLY[i], &p_SLY[i], &epsilon_SLY[i]) == 3 ) i++;	//Column order: mass density, P, energy density
	*numlinesSLY = i-1;							//set global variable to save number of lines in this file
	for (j=1; j<=i-1; j++)
	{
		p_SLY[j] = p_SLY[j]*clight*clight/p_char;				//convert to dim'less units (note P=P/c^2 in .dat files)
		epsilon_SLY[j] /= eps_char;						//convert to dim'less units
		rho_SLY[j] /= rho_char;
	}
	fclose(file);								//Close EoS file

}

extern void getRhoPts(double *rhopts)
/*
  Compute the fiducial densities: evenly spaced in the log of density
  between rho_sat and 7.4*rho_sat
*/
{
	int j;
	double rho0, rho3;

	rho0=1.0*rho_ns;	
	rho3=7.4*rho_ns;

	rhopts[1] = rho0;
	for (j=2; j<=nparam; j++) rhopts[j] = rhopts[1]*pow(10.0,log10(rho3/rhopts[1])* (j-1)/(nparam-1));
}
