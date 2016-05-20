/* C. Raithel, May 2016

 This program performs a Bayesian inversion of mass, radius observations
 to recover a parametrization of the EoS. The parametrization is our optimal
 parametrization of 5 polytopes that are spaced evenly in log(density).

 To compile: 
gcc inversion.c ../nrutil.c mtwist-1.5/mtwist.c useful_funcs.c tov_polyparam.c -o inversion -lm

*/

#include "header.h"
#include "mtwist-1.5/mtwist.h"

#define nMC 25000
#define sigma_q 5e-6

double *rhopts; //*Ppts,;							//-------------------------------------------------------//  
//double *gamma0_param, *acoef_param;					// Re-initialize GLOBAL variables defined in header file //
double *p_SLY, *epsilon_SLY, *rho_SLY;					//							 //
//double *y_m, *initRho, *rr;				   		//							 //
double *m_data, *r_data, *m_sigma, *r_sigma;				//							 //
double p_ns;								//							 //
double *initM, *initRho, *initEpsilon, *centralP;
double r_start, Pedge;
int numlinesSLY;							//-------------------------------------------------------//
							
void getRhoPts();
double integrateRhoC(double unMarg_L[]);
double findP_inSLY(double rho0);
void readinSLY(double p[], double epsilon[], double rho[]);						
double prior_P(int j,  double Ppts_local[], double gamma0_param[], double maxM);
double getPosterior(double Ppts_local[], double gamma0_param[], double mm[], 
		    double m_data[], double m_sigma[], double rr[], double r_data[], double r_sigma[]);
double gaussian(double mean, double sigma);
void revertP(double Ppts_new[], double Ppts_old[]);
double getCausalGamma(int j, double Ppts_local[]);

int main()
{

	int i,j;
	double r, posterior_old, posterior_new;
	double ratio, scriptP,step;
	double *Ppts_old, *Ppts_new;
	double *mm, *rr, *gamma0_param;

	Ppts_old = dvector(1, nparam+1);
	Ppts_new = dvector(1, nparam+1);

	gamma0_param = dvector(1,nparam);
	mm = dvector(1, nEpsilon);
	rr = dvector(1, nEpsilon);							


	rhopts = dvector(1,nparam+1);				// Allocate memory for GLOBAL variables in header file   //
	p_SLY = dvector(1, lines);			 	//							 //
	epsilon_SLY = dvector(1, lines);			//							 //
	rho_SLY = dvector(1, lines);				//							 //
	initEpsilon=dvector(1,nEpsilon);			//							 //	
	initM=dvector(1,nEpsilon);				//							 //							
	initRho = dvector(1, nEpsilon);				//							 //	
	centralP = dvector(1, nEpsilon);			//							 //	
	m_data = dvector(1,nData+1);				//							 //
	r_data = dvector(1,nData+1);				//							 //
	m_sigma = dvector(1,nData+1);				//							 //
	r_sigma = dvector(1,nData+1);				//-------------------------------------------------------//  				

        readinSLY(p_SLY,epsilon_SLY, rho_SLY);                    	//Read in the low-density data from SLY
	getRhoPts();							//Get the fiducial densities
	p_ns = findP_inSLY(rho_ns);					//Pressure at rho_saturation according to SLy 
	getMR();							//get mock MR data, dithered from SLy values
	
	Pedge=ACCURACY*pow(clight,8.0)/(Ggrav*Ggrav*Ggrav*Msolar*Msolar);	//F.'s P_edge criteria in cgs units
	Pedge/=p_char;								//F.'s P_edge criteria in my dim'less units
	r_start=r_min*Ggrav*Msolar/clight/clight/1.0e5;				//F.'s starting r_start = epsilon in my dim'less units (km/km)

	initRho[1] = 1.0*eps_min;
	for (i=2;i<=nEpsilon;i++) initRho[i] = initRho[i-1]*1.07;		//Scale starting mass density values

	for (j=1;j<=nEpsilon;j++)						//Convert initial mass densities to energy densities
	{	
		if (initRho[j] <= rhopts[1])
		{
			initEpsilon[j] = findEps0_inSLY(initRho[j]);
			centralP[j] = EOSpressure(initRho[j],0);
		}
		else	
		{		
			initEpsilon[j] = param_massToEnergy(initRho[j]);
			centralP[j] = EOSpressure(initRho[j],1);
		}
		initM[j] = r_start*r_start*r_start*initEpsilon[j];		//Initial masses enclosed by r_start = epsilon
	}


	double *test;
	test = dvector(1,7);
	test[1]=1.001;
	test[2]=0.9995;
	test[3]=0.999;
	test[4]=1.0005;
	test[5]=1.0018;
	test[6]=0.9984;
	for (i=1; i<=nparam; i++) 
		Ppts_new[i] = findP_inSLY(rhopts[i])*test[i];		//Starting Ppts = dithered from SLy values
	
	char filename[260] = "inversion_output.txt";
	FILE *f = fopen(filename,"w");
	fprintf(f,"File created by inversion.c, assuming a 5-polytrope parametrized EoS throughout the star \n");
	fprintf(f,"%d MC points used \n",nMC);
	fprintf(f,"Posterior   P_1      P_2         P_3        P_4         P_5 \n");

	int laststep_rejected = 0, toContinue;
	int accepted=0;

	tov(Ppts_new, rr, mm, gamma0_param);				//Get mass, radius, and moment of inertia for first set of Ppts
	posterior_old = getPosterior(Ppts_new,gamma0_param,mm, m_data, m_sigma,rr, r_data, r_sigma);	//Get posterior likelihood for original set of Ppts

	for (i=1; i<=nMC; i++)						//Loop through all the MC points 
	{

		if (i != 1 && laststep_rejected == 0)			//If we moved to a new step (rejected=0), use that posterior as the starting posterior
			posterior_old = posterior_new;			// (otherwise, posterior_old is still posterior_old, so we don't have to update it)


		toContinue=0;
		for (j=1; j<=nparam; j++)
		{
			Ppts_old[j] = Ppts_new[j];				//Save old Ppts to a temporary array
			step =  gaussian(0.,sigma_q);
			Ppts_new[j] += step;				//Get new set of Ppts, one Gaussian step away from old
			if (Ppts_new[j] < 0.)
				toContinue=1;
		}


		if (toContinue==1)
		{
			posterior_new = 0.;			//If any Ppt < 0, the prior will be 0 so reject and move on
			laststep_rejected = 1;
			revertP(Ppts_new, Ppts_old);
			continue;
		}

		tov(Ppts_new,rr, mm, gamma0_param);					//Get mass, radius, and I for NEW set of Ppts
		posterior_new = getPosterior(Ppts_new, gamma0_param,mm, m_data, m_sigma,rr, r_data, r_sigma);	//Get posterior likelihood for NEW set of Ppts

		ratio = posterior_new / posterior_old;
		scriptP = mt_drand();					//Mersenne Twister pseudorandom double in [0,1) with 32 bits of randomness 
		if (scriptP < ratio)					//If new P points are accepted
		{
			accepted+=1;
			laststep_rejected = 0;
			fprintf(f, "%e  %6.4e  %6.4e  %6.4e  %6.4e   %6.4e \n",posterior_new, Ppts_new[1]*p_char, Ppts_new[2]*p_char,Ppts_new[3]*p_char,Ppts_new[4]*p_char,Ppts_new[5]*p_char);
		}
		else
		{
			revertP(Ppts_new, Ppts_old);				//Otherwise, go back to previous set of Ppts
			laststep_rejected = 1;
		}
	}


	//double frac_accepted = accepted*1.0/nMC;

	fclose(f);	

	free_dvector(test,1,7);
	free_dvector(m_data,1,nData+1);
	free_dvector(r_data,1,nData+1);
	free_dvector(m_sigma,1,nData+1);
	free_dvector(r_sigma,1,nData+1);
	free_dvector(Ppts_old,1,nparam+1);
	free_dvector(Ppts_new,1,nparam+1);
	free_dvector(rho_SLY, 1, lines);
	free_dvector(epsilon_SLY, 1, lines);
	free_dvector(p_SLY, 1, lines);
	free_dvector(rr,1,nEpsilon);
	free_dvector(mm,1,nEpsilon);
	free_dvector(initRho, 1, nEpsilon);
	free_dvector(rhopts,1,nparam+1);
	//free_dvector(Ppts,1,nparam+1);
	//free_dvector(acoef_param,1,nparam);
	free_dvector(gamma0_param,1,nparam);
	free_dvector(initM,1,nEpsilon);
	free_dvector(initEpsilon,1,nEpsilon);
	free_dvector(centralP,1,nEpsilon);

	return 0;
}

void revertP(double Ppts_new[], double Ppts_old[])
{
	int j;
	for (j=1; j<=nparam; j++)
		Ppts_new[j] = Ppts_old[j];
}

double getPosterior(double Ppts_local[],  double gamma0_param[], double mm[], double m_data[], double m_sigma[], double rr[], double r_data[], double r_sigma[])
{
	int j,k;
	double likelihood, posterior, maxM;
	double *unMarg_L;
	unMarg_L = dvector(1, nEpsilon+1);


	likelihood = 1.0;					//P(data | P1, ..., P5)
	for (k=1; k<=nData; k++)
	{
		for (j=1; j<=nEpsilon; j++)
			unMarg_L[j] = exp(-(m_data[k] - mm[j])*(m_data[k] - mm[j])/(2.*m_sigma[k]*m_sigma[k]) - (r_data[k] - rr[j])*(r_data[k] - rr[j])/(2.*r_sigma[k]*r_sigma[k]) );
		likelihood *= integrateRhoC(unMarg_L);
	}



	maxM = max_array(mm,nEpsilon);				//the maximum mass achieved for this set of P1, ..., P5
	posterior = likelihood;
	for (j=1; j<=nparam; j++)
		posterior *= prior_P(j,Ppts_local, gamma0_param, maxM);			//Multiply the posterior likelihood by the priors
	
	free_dvector(unMarg_L,1, nEpsilon+1);

	return posterior;
}

double prior_P(int j, double Ppts_local[], double gamma0_param[], double maxM)
{
	double prior=1.0;

	/* Prior on P1:  P1 >= P_sat(sly), then that P1 is not allowed	*/
	if (j==1 && Ppts_local[j] < p_ns)			
		prior=0.;

	/*  Priors: P_i > P_(i-1), the gamma leading up to P_i must not be acausal, M_max >= 1.97 */
	if (j > 1 && (Ppts_local[j] < Ppts_local[j-1] || gamma0_param[j-1] > getCausalGamma(j, Ppts_local) || maxM < 1.97))  		
		prior=0.;

	return prior;
}


	
double integrateRhoC(double unMarg_L[])
/*
 For the M-R curve from a given set of (P1, ..., P5), marginalize over
 rho_c. Note, the margalization over rho_c is the same as marginalizing 
 over M, so we have:

 P(Mi, Ri | P1,... P5) = Integral P(Mi, Ri | P1,P2,P3)P_pr(rho_c) d rho_c 
 P(M, R | P1,... P5) = C*Integral_Mmin^Mmax P(M, R(M) | P1,... P5) P_pr(M) dM
 
 For now: integrate over rho_c, with flat priors in rho_c

*/ 
{	
	int j, index;
	int ngrid = 200;
	double likelihood = 0.;
	double prior_rho = 1.0;
	double deltaRho = (initRho[nEpsilon]-initRho[1])/(ngrid);
	double rho_i, m, intercept, unMarg_L_i;
	rho_i = initRho[1];
	
	for (j=1; j<=ngrid; j++)
	{
		index = bisection(rho_i, initRho, nEpsilon);		//Find the nearest value of rho_c that was actually calculated
		m = (unMarg_L[index] - unMarg_L[index+1]) /		//Calculate the slope to linearly interpolate btwn 2 calculated rho_c's 
		    (initRho[index] - initRho[index+1]);	
		intercept = unMarg_L[index] - m*initRho[index];		//Calculate y-intercept
		unMarg_L_i = m*rho_i + intercept;			//Find linear interpolation of the likelihood
	
		likelihood += unMarg_L_i*prior_rho*deltaRho;
		rho_i += deltaRho;
	}
	return likelihood;
}

double getCausalGamma(int j, double Ppts_local[])
/*Find the gamma that would correspond to a luminal EoS
  at the previous point {eps(i-1), P(i-1)}, using:
      Gamma * P/(P+eps) = (c_s)^2 = 1 
*/
{
	double eps_iMinus1, gamma;

	if (j==2)
		eps_iMinus1 = findEps0_inSLY(rhopts[1]);
	else
		eps_iMinus1 = param_massToEnergy(rhopts[j-1]);

	gamma = (eps_iMinus1 + Ppts_local[j-1])/Ppts_local[j-1] ;

	return gamma;
}


double gaussian(double mean, double sigma)
/* Draw a random number from a Gaussian distribution */
{
	double x1, x2, y1, y2;
	x1 = mt_drand();					//Mersenne Twister pseudorandom double in [0,1) with 32 bits of randomness 
	x2 = mt_drand();					
	y1 = sqrt(-2.*log(x1))*cos(2.*M_PI*x2);
	y2 = sqrt(-2.*log(x1))*sin(2.*M_PI*x2);

	return y1*sigma + mean;
}

void getRhoPts()
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


void readinSLY(double p[], double epsilon[], double rho[])
/* Read in tabulated data for the low-density regime, using the EoS Sly. */
{

	int i=1,j;
	FILE *file;
	file = fopen("/gsfs1/xdisk/craithel/sly.dat","rt");
	while ( fscanf(file,"%le %le %le", &rho[i], &p[i], &epsilon[i]) == 3 ) i++;	//Column order: mass density, P, energy density
	numlinesSLY = i-1;							//set global variable to save number of lines in this file
	for (j=1; j<=numlinesSLY; j++)
	{
		p[j] = p[j]*clight*clight/p_char;				//convert to dim'less units (note P=P/c^2 in .dat files)
		epsilon[j] /= eps_char;						//convert to dim'less units
		rho[j] /= rho_char;
	}
	fclose(file);								//Close EoS file

}

double findP_inSLY(double rho0)
/* Use bisection method to search for the energy_density that corresponds to our rho_0
   in the tabulated SLY data
*/
{

	int xmid;							//Index at midpoint
	int a=1,b=numlinesSLY;						//Index bounds for whole interval
	double m,intercept,p0;						//Slope and int for linear interpolation

	do
	{
		xmid=(a+b)/2;
		if (rho0 < rho_SLY[xmid])
			b = xmid;
		else
			a = xmid;
	} 
	while (fabs(b-a) > 1);	

	m= (p_SLY[b] - p_SLY[a]) / (rho_SLY[b] - rho_SLY[a]);	//Calculate slope to linearly interpolate btwn a and b
	intercept = p_SLY[a] - m*rho_SLY[a];				//Calculate y-intercept
	p0 = m*rho0 + intercept;
	return p0;
}




