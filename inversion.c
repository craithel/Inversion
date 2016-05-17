/* C. Raithel, May 2016

 This program performs a Bayesian inversion of mass, radius observations
 to recover a parametrization of the EoS. The parametrization is our optimal
 parametrization of 5 polytopes that are spaced evenly in log(density).

 To compile: 
gcc inversion.c ../nrutil.c mtwist-1.5/mtwist.c useful_funcs.c tov_polyparam.c -o inversion -lm

*/

#include "header.h"
#include "mtwist-1.5/mtwist.h"

#define nMC 1000
#define sigma_q 5e-6
#define nData 4

double (*Ppts);								//-------------------------------------------------------//  
double (*rhopts);							// Re-initialize GLOBAL variables defined in header file //
double (*gamma0_param);							//							 //
double (*acoef_param);							//							 //
double (*p_SLY);							//							 //
double (*epsilon_SLY);							//							 //
double (*rho_SLY);							//							 //
double (*y_m);						   		//							 //
double (*inertia);							//							 //	
double (*rr);    					   		//							 //
double p_ns;								//							 //
int numlinesSLY;							//-------------------------------------------------------//
							
double *allP1, *allP2, *allP3, *allP4, *allP5, *allPosts;		//accepted values of Ppts and corresponding posteriors

void getRhoPts();
double integrateM(double unMarg_L[]);
double findP_inSLY(double rho0);
void readinSLY(double p[], double epsilon[], double rho[]);						
double prior_P(int j, double maxM);
double getPosterior(double m_data[], double m_sigma[], double r_data[], double r_sigma[]);
double gaussian(double mean, double sigma);
void savevalues(double acceptedP[], double posterior, int iMC);
void revertP(double Ppts_old[]);
double getCausalGamma(int j);

int main()
{

	int i,j;
	double r, posterior_old, posterior_new;
	double ratio, scriptP,step;
	double *Ppts_old;

	double *m_data, *r_data;
	double *m_sigma, *r_sigma;

	m_data = dvector(1,nData+1);
	r_data = dvector(1,nData+1);
	m_sigma = dvector(1,nData+1);
	r_sigma = dvector(1,nData+1);


	//double m_data[nData+1], r_data[nData+1];
	//double m_sigma[nData+1], r_sigma[nData+1];	
	m_data[1] = 1.42807;
	m_data[2] = 1.55430;
	m_data[3] = 1.67082;
	m_data[4] = 1.80639;
	m_sigma[1] = 0.1;
	m_sigma[2] = 0.1;
	m_sigma[3] = 0.1;
	m_sigma[4] = 0.1;
	r_data[1] = 11.75985;
	r_data[2] = 11.65278;
	r_data[3] = 11.51940;
	r_data[4] = 11.51940;
	r_sigma[1] = 0.5;
	r_sigma[2] = 0.5;
	r_sigma[3] = 0.5;
	r_sigma[4] = 0.5;

	allP1 = dvector(1,nMC+1);
	allP2 = dvector(1,nMC+1);
	allP3 = dvector(1,nMC+1);
	allP4 = dvector(1,nMC+1);
	allP5 = dvector(1,nMC+1);
	allPosts = dvector(1,nMC+1);
	Ppts_old = dvector(1, nparam+1);

	Ppts = dvector(1,nparam+1);				//-------------------------------------------------------//  
	rhopts = dvector(1,nparam+1);				// Allocate memory for GLOBAL variables in header file   //
	gamma0_param = dvector(1,nparam);			//							 //						
	acoef_param = dvector(1,nparam);			//							 //							
	p_SLY = dvector(1, lines);			 	//							 //
	epsilon_SLY = dvector(1, lines);			//							 //
	rho_SLY = dvector(1, lines);				//							 //
	rr = dvector(1,nEpsilon);				//							 //			
	y_m = dvector(1,nEpsilon);				//							 //				
	inertia = dvector(1,nEpsilon);				//-------------------------------------------------------//  					


        readinSLY(p_SLY,epsilon_SLY, rho_SLY);                    	//Read in the low-density data from SLY
	getRhoPts();							//Get the fiducial densities
	double *test;
	test = dvector(1,7);
	test[1]=1.001;
	test[2]=0.9995;
	test[3]=0.999;
	test[4]=1.0005;
	test[5]=1.0018;
	test[6]=0.9984;
	for (i=1; i<=nparam; i++) 
		Ppts[i] = findP_inSLY(rhopts[i])*test[i];			//Starting Ppts = SLy values
	
	//char filename[260] = "inversion_output.txt";
	//FILE *f = fopen(filename,"w");
	//fprintf(f,"File created by inversion.c, assuming a 5-polytrope parametrized EoS throughout the star \n");
	//fprintf(f,"%d MC points used \n",nMC);
	//fprintf(f,"Posterior   P_1      P_2         P_3        P_4         P_5 \n");


	printf("File created by inversion.c, assuming a 5-polytrope parametrized EoS throughout the star \n");
	printf("%d MC points used \n",nMC);
	printf("Posterior   P_1      P_2         P_3        P_4         P_5 \n");

	int laststep_rejected = 0, toContinue;
	int accepted=0;
	for (i=1; i<=nMC; i++)						//Loop through all the MC points 
	{

		if (i==1)
		{
			tov();								//Get mass, radius, and moment of inertia for first set of Ppts
			posterior_old = getPosterior(m_data, m_sigma, r_data, r_sigma);	//Get posterior likelihood for original set of Ppts
		}

		if (i != 1 && laststep_rejected == 0)			//If we moved to a new step (rejected=0), use that posterior as the starting posterior
			posterior_old = posterior_new;			// (otherwise, posterior_old is still posterior_old, so we don't have to update it)

		toContinue=0;
		for (j=1; j<=nparam; j++)
		{
			Ppts_old[j] = Ppts[j];				//Save old Ppts to a temporary array
			step =  gaussian(0.,sigma_q);
			Ppts[j] += step;				//Get new set of Ppts, one Gaussian step away from old
			if (Ppts[j] < 0.)
				toContinue=1;
		}
		if (toContinue==1)
		{
			posterior_new = 0.;			//If any Ppt < 0, the prior will be 0 so reject and move on
			laststep_rejected = 1;
			revertP(Ppts_old);
			continue;
		}

		tov();								//Get mass, radius, and I for NEW set of Ppts
		posterior_new = getPosterior(m_data, m_sigma, r_data, r_sigma);	//Get posterior likelihood for NEW set of Ppts

		ratio = posterior_new / posterior_old;
		scriptP = mt_drand();					//Mersenne Twister pseudorandom double in [0,1) with 32 bits of randomness 
		if (scriptP < ratio)					//If new P points are accepted
		{
			accepted+=1;
			savevalues(Ppts, posterior_new, accepted);	//save the new set of Ppts and their corresponding likelihood	
			laststep_rejected = 0;
			
			//fprintf(f,"%6.4f  %6.4e  %6.4e  %6.4e  %6.4e   %6.4e \n",allPosts[accepted],allP1[accepted]*p_char, allP2[accepted]*p_char,allP3[accepted]*p_char,allP4[accepted]*p_char,allP5[accepted]*p_char);
			printf("%6.4f  %6.4e  %6.4e  %6.4e  %6.4e   %6.4e \n",allPosts[accepted],allP1[accepted]*p_char, allP2[accepted]*p_char,allP3[accepted]*p_char,allP4[accepted]*p_char,allP5[accepted]*p_char);
		}
		else
		{
			revertP(Ppts_old);				//Otherwise, go back to previous set of Ppts
			laststep_rejected = 1;
		}
	}


	//double frac_accepted = accepted*1.0/nMC;

	//fclose(f);	
	//printf("Saved to file %s\n", filename);

	free_dvector(test,1,7);
	free_dvector(m_data,1,nData+1);
	free_dvector(r_data,1,nData+1);
	free_dvector(m_sigma,1,nData+1);
	free_dvector(r_sigma,1,nData+1);
	free_dvector(Ppts_old,1,nparam+1);
	free_dvector(allP1, 1, nMC+1);
	free_dvector(allP2, 1, nMC+1);
	free_dvector(allP3, 1, nMC+1);
	free_dvector(allP4, 1, nMC+1);
	free_dvector(allP5, 1, nMC+1);
	free_dvector(allPosts, 1, nMC+1);
	free_dvector(rho_SLY, 1, lines);
	free_dvector(epsilon_SLY, 1, lines);
	free_dvector(p_SLY, 1, lines);
	free_dvector(rr,1,nEpsilon);
	free_dvector(y_m,1,nEpsilon);
	free_dvector(inertia, 1, nEpsilon);
	free_dvector(rhopts,1,nparam+1);
	free_dvector(Ppts,1,nparam+1);
	free_dvector(acoef_param,1,nparam);
	free_dvector(gamma0_param,1,nparam);

	return 0;
}

void savevalues(double acceptedP[], double posterior, int iMC)
{
	allP1[iMC] = acceptedP[1];
	allP2[iMC] = acceptedP[2];
	allP3[iMC] = acceptedP[3];
	allP4[iMC] = acceptedP[4];
	allP5[iMC] = acceptedP[5];
	allPosts[iMC] = posterior;
}  

void revertP(double Ppts_old[])
{
	int j;
	for (j=1; j<=nparam; j++)
		Ppts[j] = Ppts_old[j];
}

double getPosterior(double m_data[], double m_sigma[], double r_data[], double r_sigma[])
{
	int j,k;
	double likelihood, posterior, maxM;
	double *unMarg_L;
	unMarg_L = dvector(1, nEpsilon+1);

	likelihood = 1.0;					//P(data | P1, ..., P5)
	for (k=1; k<=nData; k++)
	{
		for (j=1; j<=nEpsilon; j++)
			unMarg_L[j] = exp(-(m_data[k] - y_m[j])*(m_data[k] - y_m[j])/(2.*m_sigma[k]*m_sigma[k]) - (r_data[k] - rr[j])*(r_data[k] - rr[j])/(2.*r_sigma[k]*r_sigma[k]) );
		likelihood *= integrateM(unMarg_L);
	}

	maxM = max_array(y_m,nEpsilon);				//the maximum mass achieved for this set of P1, ..., P5
	posterior = likelihood;
	for (j=1; j<=nparam; j++)
		posterior *= prior_P(j,maxM);			//Multiply the posterior likelihood by the priors
	
	free_dvector(unMarg_L,1, nEpsilon+1);

	return posterior;
}

double prior_P(int j, double maxM)
{
	double causal_g = getCausalGamma(j); 
	double prior=1.0;

	/* Prior on P1:  P1 >= P_sat(sly), then that P1 is not allowed	*/
	if (j==1 && Ppts[j] < p_ns)			
		prior=0.;

	/*  Priors: P_i > P_(i-1), the gamma leading up to P_i must not be acausal, M_max >= 1.97 */
	if (j > 1 && (Ppts[j] < Ppts[j-1] || gamma0_param[j-1] > causal_g || maxM < 1.97))  		
		prior=0.;

	return prior;
}


	
double integrateM(double unMarg_L[])
/*
 For the M-R curve from a given set of (P1, ..., P5), marginalize over
 rho_c. Note, the margalization over rho_c is the same as marginalizing 
 over M, so we have:

 P(Mi, Ri | P1,... P5) = Integral P(Mi, Ri | P1,P2,P3)P_pr(rho_c) d rho_c 
 P(M, R | P1,... P5) = C*Integral_Mmin^Mmax P(M, R(M) | P1,... P5) P_pr(M) dM
 
*/ 
{	
	int j;
	double lowerM, upperM;
	double lowerM_index, upperM_index;
	double likelihood = unMarg_L[1];
	double prior_M = 1.0;

	lowerM = min_array(y_m, nEpsilon);
	upperM = max_array(y_m, nEpsilon);

	if (lowerM > 0.1)	
	{
		printf("warning: min M = %f > 0.1\n Terminating program.\n",lowerM);
		exit(0); 
	} 

	lowerM_index = bisection(0.1, y_m, nEpsilon);				//Find the index corresponding to M=0.1
	upperM_index = bisection(upperM, y_m, nEpsilon);			//Find the index corresponding to Mmax

	for (j=lowerM_index; j<=upperM_index-1;j++)
		likelihood += unMarg_L[j+1]*(y_m[j+1]-y_m[j])*prior_M;

	return likelihood;
}

double getCausalGamma(int j)
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

	gamma = (eps_iMinus1 + Ppts[j-1])/Ppts[j-1] ;

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




