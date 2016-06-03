/* C. Raithel, May 2016

 This program performs a Bayesian inversion of mass, radius observations
 to recover a parametrization of the EoS. The parametrization is our optimal
 parametrization of 5 polytopes that are spaced evenly in log(density).

 To compile: 
gcc inversion.c ../nrutil.c mtwist-1.5/mtwist.c useful_funcs.c tov_polyparam.c mockMR.c -o inversion -lm

*/

#include "tov.h"
#include "mockMR.h"
#include "mtwist-1.5/mtwist.h"
#include "mpi.h"
#include "initP.h"

#define nMC 7400000							//Total number of desired MC steps; to be split among N parallel processes
#define nBurn 0								//Number of accepted steps to ignore
#define sigma_q 7e-6							//Step size (steps drawn from a gaussian distro with mean 0 and sigma_q)
#define nrand 7								//Number of random numbers to be used in each loop of each process
#define REQUEST 1
#define REPLY 2

double *rhopts; 							//-------------------------------------------------------//  
double *p_SLY, *epsilon_SLY, *rho_SLY;					// Re-initialize GLOBAL variables defined in header file //
double *m_data, *r_data, *m_sigma, *r_sigma;				//							 //
double *initM, *initRho, *initEpsilon, *centralP;			//							 //
double r_start, Pedge;							//							 //
int numlinesSLY;							//-------------------------------------------------------//
double p_ns;						

double integrateRhoC(double unMarg_L[]);
double prior_P(int j,  double Ppts_local[], double gamma0_param[], double acoef_param[], double maxM);
double getPosterior(double Ppts_local[], double gamma0_param[],double acoef_param[], double mm[], 
		    double m_data[], double m_sigma[], double rr[], double r_data[], double r_sigma[]);
void gaussian(double *steps, double mean, double sigma1, double sigma2, double sigma3, double sigma4, double sigma5, double *rands);
void revertP(double Ppts_new[], double Ppts_old[]);
double getCausalGamma(int j, double Ppts_local[], double gamma0_param[], double acoef_param[]);

int main( int argc, char *argv[])
{

	int i,j, nMC_thisrank;
	int laststep_rejected = 0, toContinue;
	int accepted=0;
	int attempted=0;
	double r, posterior_old, posterior_new;
	double ratio, scriptP,step;
	double *Ppts_old, *Ppts_new;
	double *mm, *rr, *gamma0_param, *acoef_param;
	double *test, *steps;
	double mean_P1, mean_P2, mean_P3, mean_P4, mean_P5;	
	double sigma_P1, sigma_P2, sigma_P3, sigma_P4, sigma_P5;
	
	int ranks[1], numprocs, myid, server, workerid;
	int request;
	double rands[nrand+1];
		
	MPI_Comm world, workers;
	MPI_Group world_group, worker_group;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	world = MPI_COMM_WORLD;
	MPI_Comm_size(world, &numprocs);
	MPI_Comm_rank(world, &myid);
	server = numprocs-1;				 			//last proc is server for random numbers
	
	p_SLY = dvector(1, lines);			 	
	epsilon_SLY = dvector(1, lines);			
	rho_SLY = dvector(1, lines);				
	m_data = dvector(1,nData+1);				
	r_data = dvector(1,nData+1);				
	m_sigma = dvector(1,nData+1);				
	r_sigma = dvector(1,nData+1);		
	if (myid==0)								//In main process only,
	{
		getMR();							// get mock MR data (dithered from SLy values)
		readinSLY(p_SLY,epsilon_SLY, rho_SLY, &numlinesSLY);          	//Read in the low-density data from SLY
	}									// and broadcast to all other processes
	MPI_Bcast(m_data, nData+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Bcast(r_data, nData+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_sigma, nData+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(r_sigma, nData+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(p_SLY, lines, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(epsilon_SLY, lines, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(rho_SLY, lines, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numlinesSLY, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Comm_group(world, &world_group);						
	ranks[0] = server;							//Create a rand server by excluding one communicator
	MPI_Group_excl(world_group, 1, ranks, &worker_group);			// from the "world" group. The rand server supplies random nums
	MPI_Comm_create(world, worker_group, &workers);				// to all other processes. "Worker" is the communicator
	MPI_Group_free(&worker_group);						// group with all other processes that do the brunt of the code.
										
	if (myid == server)							// Rand server
	{
		do
		{
			MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status);
			if (request)
			{
				for (i=1;i<=nrand;i++) rands[i] = mt_drand(); 
				MPI_Send(rands, nrand+1, MPI_DOUBLE, status.MPI_SOURCE, REPLY, world);
			}

		}
		while (request > 0);
	}
	else									// Worker process
	{
		request = 1;

		MPI_Comm_rank(workers, &workerid);				//Determine rank

		rhopts = dvector(1,nparam);				// Allocate memory for GLOBAL variables in header file   //
		initEpsilon=dvector(1,nEpsilon);			//							 //	
		initM=dvector(1,nEpsilon);				//							 //							
		initRho = dvector(1, nEpsilon);				//							 //	
		centralP = dvector(1, nEpsilon);			//-------------------------------------------------------// 	

		Ppts_old = dvector(1, nparam);
		Ppts_new = dvector(1, nparam);
		steps = dvector(1, nparam);
		gamma0_param = dvector(1,nparam-1);
		acoef_param = dvector(1,nparam-1);
		mm = dvector(1, nEpsilon);
		rr = dvector(1, nEpsilon);							

		p_ns = bisect_linint(rho_ns, rho_SLY, p_SLY, numlinesSLY);				//Pressure at rho_saturation according to SLy 
		getRhoPts(rhopts);								//Get the fiducial densities
		//test = dvector(1,7);	

		if (myid==0)
		{
			initP(&mean_P1, &sigma_P1, &mean_P2, &sigma_P2, &mean_P3, &sigma_P3, &mean_P4, &sigma_P4, &mean_P5, &sigma_P5); 
			exit(0);
		}

		int test_prior = 0;
		while (test_prior == 0)
		{
			MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);		//Send request for random numbers
			MPI_Recv(rands, nrand+1, MPI_DOUBLE, server, REPLY, world, &status); //Receive 7 random numbers

			/*test[1]=1.;			//P_ns is fixed and should not be varied
			test[2]=rands[1]*3.;
			test[3]=rands[2]*3.;
			test[4]=rands[3]*3.;
			test[5]=rands[4]*3.;
			test[6]=rands[5]*3.;
			for (i=1; i<=nparam; i++) 
				Ppts_new[i] = bisect_linint(rhopts[i], rho_SLY, p_SLY, numlinesSLY)*test[i];	//Starting Ppts = dithered from SLy values
			*/

			Ppts_new[1] = p_ns;
			Ppts_new[2] = mean_P1*(rands[1]*3.);
			Ppts_new[3] = mean_P1*(rands[2]*3.);
			Ppts_new[4] = mean_P1*(rands[3]*3.);
			Ppts_new[5] = mean_P1*(rands[4]*3.);
			Ppts_new[6] = mean_P1*(rands[5]*3.);

			getGammaA(Ppts_new, gamma0_param, acoef_param);

			test_prior=1;
			if (Ppts_new[2] < p_ns || Ppts_new[3] < 2.85e-5)	// Prior on P1 and P2:  P1 >= P_sat(sly), P2 >= 7.56MeV/fm^3
				test_prior = 0;			
			else
			{
				for (j=2; j<=nparam; j++)
				{
					if (Ppts_new[j] < Ppts_new[j-1] || gamma0_param[j-1] > getCausalGamma(j, Ppts_new,gamma0_param,acoef_param) )  		
					{
						test_prior=0;
						continue;
					}
				}
			}
		}

		edgeConditions(&r_start, &Pedge);
		initConditions(initM, initRho, initEpsilon, centralP, Ppts_new, gamma0_param, acoef_param);

		char filename[260];
		sprintf(filename,"chain_output/inversion_output_%d.txt",myid);
		FILE *f = fopen(filename,"w");
		fprintf(f,"File created by inversion.c, assuming a 5-polytrope parametrized EoS throughout the star (P0 = P_sat) \n");
		fprintf(f,"Starting pressures from histogram of 42 EoS and dithered by: %f %f %f %f %f\n", rands[1]*3., rands[2]*3., rands[3]*3., rands[4]*3., rands[5]*3.);
		fprintf(f,"N_MC and Acceptance rate details in footer. \n");
		fprintf(f,"Posterior      P_1         P_2          P_3         P_4          P_5 \n");
		fflush(f);

		tov(Ppts_new, rr, mm, gamma0_param, acoef_param);			  //Get mass, radius, and update gamma and 'a' coefs for first set of Ppts
		posterior_old = getPosterior(Ppts_new,gamma0_param,acoef_param,		  //Get posterior likelihood for original set of Ppts
 					     mm, m_data, m_sigma,rr, r_data, r_sigma);

		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);		//Send request for random numbers

		i=1;
		nMC_thisrank = nMC/(numprocs-1);
		while (i <= nMC_thisrank)						   //Loop through all MC points
		{
			MPI_Recv(rands, nrand+1, MPI_DOUBLE, server, REPLY, world, &status); //Receive 3 random numbers for this loop iteration

			if (i != 1 && laststep_rejected == 0)				   //If we moved to a new step (rejected=0), use that posterior as the starting posterior
				posterior_old = posterior_new;				   // (otherwise, posterior_old is still posterior_old, so we don't have to update it)


			toContinue=0;
			gaussian(steps, 0.,sigma_P1, sigma_P2, sigma_P3, sigma_P4_sigma_P5, rands);
			for (j=2; j<=nparam; j++)					//Note: P0 is fixed, so we don't take steps in it
			{
				Ppts_old[j] = Ppts_new[j];				   //Save old Ppts to a temporary array
				Ppts_new[j] += steps[j-1];			      	   //Get new set of Ppts, one Gaussian step away from old
				if (Ppts_new[j] < p_ns)
					toContinue=1;
			}


			if (toContinue==1)
			{
				attempted++;
				posterior_new = 0.;					   //If any Ppt < 0, the prior will be 0 so reject and move on
				laststep_rejected = 1;
				revertP(Ppts_new, Ppts_old);
				MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
				continue;
			}


			initConditions(initM, initRho, initEpsilon, centralP, Ppts_new, gamma0_param, acoef_param); //Get new init values for new set of Ppts

			tov(Ppts_new,rr, mm, gamma0_param,acoef_param);			   //Get M,R and update gamma and a_coef for NEW set of Ppts
			posterior_new = getPosterior(Ppts_new, gamma0_param,acoef_param,   //Get posterior likelihood for NEW set of Ppts
						    mm, m_data, m_sigma,rr, r_data, r_sigma);

			ratio = posterior_new / posterior_old;
			scriptP = rands[7];						   //MT pseudorandom double in [0,1) with 32 bits of randomness 
			
			if (scriptP < ratio)						   //If new P points are accepted
			{
				laststep_rejected = 0;
				//if (i > nBurn)						   //Save PPts if we're past the burn-in period
				//{
				fprintf(f, "%e  %6.4e  %6.4e  %6.4e  %6.4e   %6.4e \n",posterior_new, Ppts_new[2]*p_char, Ppts_new[3]*p_char,Ppts_new[4]*p_char,Ppts_new[5]*p_char,Ppts_new[6]*p_char);
				fflush(f);
				accepted++;
				//}
				i++;							 
			}
			else
			{
				revertP(Ppts_new, Ppts_old);				//Otherwise, go back to previous set of Ppts
				laststep_rejected = 1;
			}
			attempted++;
			if (i <= nMC_thisrank)
				request=1;
			else
				request=0;
			
			if (myid==0)
				MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
			else
			{
				if (request)
					MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
			}

		}

		fprintf(f,"Total steps attempted in this process: %d nBurnIn(number accepted that are thrown out): %d \n", attempted, nBurn);
		fprintf(f,"Total steps attempted after BurnIN: %d  Number accepted after BurnIn: %d \n", attempted-nBurn, accepted);
		fclose(f);	

		//free_dvector(test,1,7);
		free_dvector(Ppts_old,1,nparam);
		free_dvector(Ppts_new,1,nparam);
		free_dvector(steps,1,nparam);
		free_dvector(rhopts,1,nparam);
		free_dvector(rho_SLY, 1, lines);
		free_dvector(epsilon_SLY, 1, lines);
		free_dvector(p_SLY, 1, lines);
		free_dvector(rr,1,nEpsilon);
		free_dvector(mm,1,nEpsilon);
		free_dvector(acoef_param,1,nparam-1);
		free_dvector(gamma0_param,1,nparam-1);
		free_dvector(initM,1,nEpsilon);
		free_dvector(initEpsilon,1,nEpsilon);
		free_dvector(centralP,1,nEpsilon);
		free_dvector(initRho, 1, nEpsilon);
		
		MPI_Comm_free(&workers);	

	}
	
	free_dvector(m_data,1,nData+1);
	free_dvector(r_data,1,nData+1);
	free_dvector(m_sigma,1,nData+1);
	free_dvector(r_sigma,1,nData+1);
	MPI_Finalize();
	return 0;
}

void revertP(double Ppts_new[], double Ppts_old[])
/* Go back to the old values of Ppts if a new step was rejected */
{
	int j;
	for (j=2; j<=nparam; j++)
		Ppts_new[j] = Ppts_old[j];
}

double getPosterior(double Ppts_local[],  double gamma0_param[], double acoef_param[], double mm[], double m_data[], double m_sigma[], double rr[], double r_data[], double r_sigma[])
/* Get the posterior likelihood for a set of Ppts using
 	P(P1,...,P5 | data) ~ P(data | P1,...,P5) * [P_pr(P1)* ... *P_pr(P5)] 	  
 */
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
	if (Ppts_local[2] < p_ns || Ppts_local[3] < 2.85e-5)	// Prior on P1 and P2:  P1 >= P_sat(sly), P2 >= 7.56MeV/fm^3
		posterior = 0.;					
	else
	{
		for (j=2; j<=nparam; j++)
			posterior *= prior_P(j,Ppts_local, gamma0_param,acoef_param, maxM);			//Multiply the posterior likelihood by the priors
	}
	free_dvector(unMarg_L,1, nEpsilon+1);

	return posterior;
}

double prior_P(int j, double Ppts_local[], double gamma0_param[], double acoef_param[], double maxM)
/* Assume flat priors in pressure. A given set of Ppts must obey:
    1. M_max > 1.9 M_sun
    2. P1 >= P_sat
    3. P_i >= P_(i-1)
    4. Gamma_i <= Gamma_luminal
    5. P2 <= 7.56 MeV/fm^3 (from Ozel et al 2016 prior on their P1 -- NEEDS TO BE UPDATED FOR OUR P1)
 If any of these are broken, the prior becomes 0. 
*/
{
	double prior=1.0;

	/* Prior on P1:  P1 >= P_sat(sly), then that P1 is not allowed	*/
	//if (j==1 && Ppts_local[j] < p_ns)			
	//	prior=0.;

	//if (j==2 &&  Ppts_local[j] > 2.85e-5)
	//	prior=0.;

	/*  Priors: P_i > P_(i-1), the gamma leading up to P_i must not be acausal, M_max >= 1.97 */
	if (Ppts_local[j] < Ppts_local[j-1] || gamma0_param[j-1] > getCausalGamma(j, Ppts_local,gamma0_param,acoef_param) || maxM < 1.97)  		
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

double getCausalGamma(int j, double Ppts_local[], double gamma0_param[], double acoef_param[])
/*Find the gamma that would correspond to a luminal EoS
  at the previous point {eps(i-1), P(i-1)}, using:
      Gamma * P/(P+eps) = (c_s)^2 = 1 
*/
{
	double eps_iMinus1, gamma;

	if (j==2)
		eps_iMinus1 = findEps0_inSLY(rhopts[1]);
	else
		eps_iMinus1 = param_massToEnergy(rhopts[j-1],Ppts_local, gamma0_param, acoef_param);

	gamma = (eps_iMinus1 + Ppts_local[j-1])/Ppts_local[j-1] ;

	return gamma;
}


void gaussian(double *steps, double mean, double sigma1, double sigma2, double sigma3, double sigma4, double sigma5, double *rands)
/* Draw a random number from a Gaussian distribution */
{
	steps[1] = sqrt(-2.*log(rands[1]))*cos(2.*M_PI*rands[2])*sigma1 + mean;
	steps[2] = sqrt(-2.*log(rands[1]))*sin(2.*M_PI*rands[2])*sigma2 + mean;

	steps[3] = sqrt(-2.*log(rands[3]))*cos(2.*M_PI*rands[4])*sigma3 + mean;
	steps[4] = sqrt(-2.*log(rands[3]))*sin(2.*M_PI*rands[4])*sigma4 + mean;

	steps[5] = sqrt(-2.*log(rands[5]))*cos(2.*M_PI*rands[6])*sigma5 + mean;

}

