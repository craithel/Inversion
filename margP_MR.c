/* C. Raithel, May 2016
  
   This program finds the best fit values of (P2, ..., P5) from the MARGINALIZED output
   of inversion.c by searching the histogrammed values of each P and finding all values
   within 1/e of the max.
   
   It then runs the parametric version of the TOV solver to get the M,R curve 
   corresponding to each set of most likely (P2, ..., P5) and outputs these allowed
   M,R points to a single file: MR_marg_1overE.txt  or MR_marg_1overSqrtE.txt
 
   To compile:
   icc margP_MR.c ../nrutil.c useful_funcs.c tov_helper.c tov_polyparam.c -o margP_MR -lm

  */

#include "../nrutil.h"
#include "header_useful.h"
#include "tov.h"
#include "mtwist-1.5/mtwist.h"
#include "mpi.h"

#define nMC 5000
#define nMCperfile 5000
#define nrand 2
#define nBins 50

#define REQUEST 1
#define REPLY 2

double *rhopts; 							//-------------------------------------------------------//  
double *p_SLY, *epsilon_SLY, *rho_SLY;					// Re-initialize GLOBAL variables defined in header file //
double *initM, *initRho, *initEpsilon, *centralP;			//							 //
double r_start, Pedge, p_ns;						//							 //
int numlinesSLY;							//-------------------------------------------------------//

void getBins(double *P, double *bins, double *bins_centered);
double drawFromDistro(double Pbelow, double *bins, double *bins_centered, int *hist, int myid,  int server, MPI_Comm world, MPI_Status status);
double checkPriors(double *Ppts, double *gamma0_param, double *acoef_param, double maxM);
double getCausalGamma(int j, double Ppts_local[], double gamma0_param[], double acoef_param[]);

int main(int argc, char *argv[])
{
	int i,j,k;
	double *P1, *P2, *P3, *P4, *P5;
	double *P1_bins, *P2_bins, *P3_bins, *P4_bins, *P5_bins;
	double *P1_bins_c, *P2_bins_c, *P3_bins_c, *P4_bins_c, *P5_bins_c;
	int *P1_hist, *P2_hist, *P3_hist, *P4_hist, *P5_hist;
	double *Ppts, *gamma0_param, *acoef_param;
	double *rr, *mm;
	double maxM;
	char buff[256];
	char post[256];
	FILE *file;

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

	initRho = dvector(1, nEpsilon);
	initEpsilon = dvector(1, nEpsilon);
	initM = dvector(1, nEpsilon);
	centralP = dvector(1, nEpsilon);

	rhopts = dvector(1, nparam);
	p_SLY = dvector(1, lines);
	rho_SLY = dvector(1, lines);
	epsilon_SLY = dvector(1, lines);

	P1 = dvector(1,nMC);
	P2 = dvector(1,nMC);
	P3 = dvector(1,nMC);
	P4 = dvector(1,nMC);
	P5 = dvector(1,nMC);
	
	P1_bins = dvector(1, nBins);
	P2_bins = dvector(1, nBins);
	P3_bins = dvector(1, nBins);
	P4_bins = dvector(1, nBins);
	P5_bins = dvector(1, nBins);

	P1_bins_c = dvector(1, nBins);
	P2_bins_c = dvector(1, nBins);
	P3_bins_c = dvector(1, nBins);
	P4_bins_c = dvector(1, nBins);
	P5_bins_c = dvector(1, nBins);

	P1_hist = ivector(1,nBins-1);
	P2_hist = ivector(1,nBins-1);
	P3_hist = ivector(1,nBins-1);
	P4_hist = ivector(1,nBins-1);
	P5_hist = ivector(1,nBins-1);

	Ppts = dvector(1,nparam);
	gamma0_param = dvector(1,nparam-1);
	acoef_param = dvector(1, nparam-1);
	rr = dvector(1, nEpsilon);
	mm = dvector(1, nEpsilon);

	if (myid==0)
	{

		char infile[256];
		i=1;
		//for (j=0; j<=368; j++)
		//{
		sprintf(infile,"chain_output/inversion_output_%d.txt",36);
		file = fopen(infile, "rt");
		fgets(buff, 256, file);
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	

		for (k=1; k<=nMCperfile; k++)
		{
			fscanf(file, "%s %le %le %le %le %le", &post, &P1[i], &P2[i], &P3[i], &P4[i], &P5[i]);
			P1[i] /= p_char;
			P2[i] /= p_char;
			P3[i] /= p_char;
			P4[i] /= p_char;
			P5[i] /= p_char;
			i++;
		}
		fclose(file);
		//}

	}

	MPI_Bcast(P1, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P2, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P3, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P4, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P5, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
	getBins(P1, P1_bins, P1_bins_c);
	getBins(P2, P2_bins, P2_bins_c);
	getBins(P3, P3_bins, P3_bins_c);
	getBins(P4, P4_bins, P4_bins_c);
	getBins(P5, P5_bins, P5_bins_c);

	for (i=1; i<=nBins-1; i++)
	{
		for (j=1; j<=nMC; j++)
		{
			if (P1[j] >= P1_bins[i] && P1[j] < P1_bins[i+1]) P1_hist[i] +=1;
			if (P2[j] >= P2_bins[i] && P2[j] < P2_bins[i+1]) P2_hist[i] +=1;
			if (P3[j] >= P3_bins[i] && P3[j] < P3_bins[i+1]) P3_hist[i] +=1;
			if (P4[j] >= P4_bins[i] && P4[j] < P4_bins[i+1]) P4_hist[i] +=1;
			if (P5[j] >= P5_bins[i] && P5[j] < P5_bins[i+1]) P5_hist[i] +=1;
		}
	}

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

		edgeConditions(&r_start, &Pedge);
		readinSLY(p_SLY, epsilon_SLY, rho_SLY, &numlinesSLY);
		getRhoPts(rhopts);	
		p_ns = bisect_linint(rho_ns, rho_SLY, p_SLY, numlinesSLY);

		char filename[256];
		sprintf(filename,"margMR_output/MR_marg_%d.txt",myid);
		FILE *f = fopen(filename,"w");
		fprintf(f, "MR values for a randomly selected set of P's from the 5 MARGINALIZED pressure distributions\n");
		fprintf(f, "The set of 5 random P's was checked to ensure they obeyed all our priors\n");
		fflush(f);

		int accepted=0;
		while (accepted ==0)
		{
			Ppts[1] = p_ns;
			Ppts[2] = drawFromDistro(Ppts[1], P1_bins, P1_bins_c, P1_hist, myid, server, world, status);
			Ppts[3] = drawFromDistro(Ppts[2], P2_bins, P2_bins_c, P2_hist, myid, server, world, status);
			Ppts[4] = drawFromDistro(Ppts[3], P3_bins, P3_bins_c, P3_hist, myid, server, world, status);
			Ppts[5] = drawFromDistro(Ppts[4], P4_bins, P4_bins_c, P4_hist, myid, server, world, status);
			Ppts[6] = drawFromDistro(Ppts[5], P5_bins, P5_bins_c, P5_hist, myid, server, world, status);

			initConditions(initM, initRho, initEpsilon, centralP, Ppts, gamma0_param, acoef_param);
			tov(Ppts, rr, mm,gamma0_param, acoef_param);

			maxM = max_array(mm, nEpsilon);
			accepted = checkPriors(Ppts, gamma0_param, acoef_param, maxM );
			
			if (accepted==0)
				fprintf(f, "rejected: maxM = %f ...  %e %e %e %e %e %e \n",maxM, Ppts[1]*p_char, Ppts[2]*p_char, Ppts[3]*p_char, Ppts[4]*p_char, Ppts[5]*p_char, Ppts[6]*p_char);
		}

		for (j=1; j<=nEpsilon; j++) fprintf(f, "%f %e \n", rr[j], mm[j]);
		fclose(f);
	}

	if (myid==0)
	{
		request=0;
		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
	}

	if (myid != server)
		MPI_Comm_free(&workers);

	free_dvector(rhopts, 1, nparam);
	free_dvector(gamma0_param, 1, nparam-1);
	free_dvector(acoef_param, 1, nparam-1);
	free_dvector(p_SLY,1,lines);
	free_dvector(rho_SLY,1,lines);
	free_dvector(epsilon_SLY,1,lines);
	free_dvector(initRho, 1, nEpsilon);
	free_dvector(initEpsilon, 1, nEpsilon);
	free_dvector(initM, 1, nEpsilon);
	free_dvector(centralP, 1, nEpsilon);
	free_dvector(rr, 1, nEpsilon);
	free_dvector(mm, 1, nEpsilon);
	free_dvector(P1, 1, nMC);
	free_dvector(P2, 1, nMC);
	free_dvector(P3, 1, nMC);
	free_dvector(P4, 1, nMC);
	free_dvector(P5, 1, nMC);
	free_dvector(Ppts, 1, nparam);
	free_dvector(P1_bins, 1, nBins);
	free_dvector(P2_bins, 1, nBins);
	free_dvector(P3_bins, 1, nBins);
	free_dvector(P4_bins, 1, nBins);
	free_dvector(P5_bins, 1, nBins);
	free_ivector(P1_hist, 1, nBins-1);
	free_ivector(P2_hist, 1, nBins-1);
	free_ivector(P3_hist, 1, nBins-1);
	free_ivector(P4_hist, 1, nBins-1);
	free_ivector(P5_hist, 1, nBins-1);

	MPI_Finalize();

	return 0;
}

void getBins(double *P, double *bins, double *bins_centered)
{
	int i;
	double lbound, ubound, delta_bin, half_delta;
	lbound = min_array(P, nMC);
	ubound = max_array(P, nMC);

	delta_bin = (ubound - lbound)/(1.*nBins);

	bins[1] = lbound - delta_bin;
	for (i=2; i<=nBins; i++) bins[i] = bins[i-1] + delta_bin;
	
	half_delta = delta_bin*0.5;	
	for (i=1; i<=nBins-1; i++) bins_centered[i] = bins[i] + half_delta;

}

double drawFromDistro(double Pbelow, double *bins, double *bins_centered,int *hist, int myid, int server, MPI_Comm world, MPI_Status status)
{
	int i, a, notAccepted;
	int request, max_hist;
	double rand_probi, rand_Pi, lower_lim;
	double rands[nrand+1];
	double random_P;

	request = 1;
	max_hist = max_iarray(hist, nBins-1);

	MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);

	notAccepted=1;
	while (notAccepted)
	{
		MPI_Recv(rands, nrand+1, MPI_DOUBLE, server, REPLY, world, &status); 	//Receive 2 random numbers

		lower_lim = max(bins[1], Pbelow);
		rand_Pi = lower_lim + rands[1]*(bins[nBins]-lower_lim);	 //Scale 1st rand to be within range of pressures but bigger than P_(i-1)		
		rand_probi = max_hist*rands[2];

		if (rand_Pi < bins[1] || rand_Pi >= bins[nBins])
		{
			MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
		}

		for (i=1; i<=nBins-1; i++)
		{
			if (rand_Pi >= bins[i] && rand_Pi < bins[i+1])		//Find which pressure bin the random number is in
			{
				a=i;
				continue;
			}
		}

		if (rand_probi <= hist[a])			//Accept the pressure if rand is within the histogram prob for that P
		{
			notAccepted = 0;
			random_P = bins_centered[a];		
			request = 0;
		}
		else
		{
			if (request)
				MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
		}
	}

	return random_P;

}
double checkPriors(double *Ppts, double *gamma0_param, double *acoef_param, double maxM)
{
	int j, prior = 1;

	if (Ppts[2] < p_ns || Ppts[3] < 2.85e-5 || maxM < 1.97) 	// Prior on P1 and P2:  P1 >= P_sat(sly), P2 >= 7.56MeV/fm^3
		prior=0;
	else
	{
		for (j=2; j<=nparam; j++)
		{
			if (Ppts[j] < Ppts[j-1] || gamma0_param[j-1] > getCausalGamma(j, Ppts,gamma0_param,acoef_param) )  		
			{
				prior=0;
				break;
			}
		}
	}

	return prior;
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
