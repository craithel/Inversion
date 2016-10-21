/* C. Raithel, May 2016

 This program performs a Bayesian inversion of mass, radius observations
 to recover a parametrization of the EoS. The parametrization is our optimal
 parametrization of 5 polytopes that are spaced evenly in log(density).

 To compile: 
mpicc -g inversion.c ../nrutil.c mtwist-1.5/mtwist.c useful_funcs.c tov_polyparam.c tov_helper.c mockMR.c initP.c -o inversion -lm

*/

#include "tov.h"
#include "mockMR.h"
#include "mtwist-1.5/mtwist.h"
#include "mpi.h"
#include "initP.h"
#include <time.h>

#define nMC 2368000 //9472000						//Total number of desired MC steps; to be split among N parallel processes
#define nMRgrid 50
#define nBurn 0								//Number of accepted steps to ignore
#define nrand 7								//Number of random numbers to be used in each loop of each process
#define REQUEST 1
#define REPLY 2

double *rhopts;						//-------------------------------------------------------//  
double *p_SLY, *epsilon_SLY, *rho_SLY;					// Re-initialize GLOBAL variables defined in header file //
double *m_data, *r_data, *m_sigma, *r_sigma;				//							 //
double moi_EOS, sigma_moi;
double *initM, *initRho, *initEpsilon, *centralP;			//							 //
double r_start, Pedge;							//							 //
int numlinesSLY;							//-------------------------------------------------------//
double p_ns;						
double *Mgrid;
double *Rgrid;

double integrateRhoC(double unMarg_L[], int maxM_index);
double prior_P(int j,  double Ppts_local[],double epspts_local[], double gamma0_param[], double acoef_param[]);
double getPosterior(double Ppts_local[], double epspts_local[], double gamma0_param[],double acoef_param[], double mm[], 
		    double m_data[], double m_sigma[], double rr[], double r_data[], double r_sigma[], double inertia[]);
void gaussian(double *steps, double mean, double *sigmas, double *rands);
void revertP(double Ppts_new[], double Ppts_old[]);
double getCausalGamma(int j, double Ppts_local[], double gamma0_param[], double acoef_param[]);
double getCs(int i, double Ppts_local[], double epspts_local[] );
void getMRhist(int myid, double *mm, double *rr, int **hist);
int firstmax(double *array, int length);
double integrateM(double unMarg_L[], int unstable1, int unstable2, double deltaM, double mgrid[], double rgrid[], double sigmaR, double sigmaM);
double getLikelihood(double sigmaM, double sigmaR, double dataM, double dataR, double EOSm, double EOSr);

int main( int argc, char *argv[])
{

	int i,j,k,l, nMC_thisrank;
	int laststep_rejected = 0, toContinue;
	int accepted=0;
	int attempted=0;
	int realization;
	double r, posterior_old, posterior_new;
	double ratio, scriptP,step;
	double *epspts;
	double *Ppts_old, *Ppts_new;
	double *mm, *rr, *gamma0_param, *acoef_param;
	double *steps;
	double mean_P1, mean_P2, mean_P3, mean_P4, mean_P5;	
	double sigma_P1, sigma_P2, sigma_P3, sigma_P4, sigma_P5;
	double *sigmaPs;
	double maxM, firstMax;
	double *inertia;
	inertia = (double*)malloc((nEpsilon+1)*sizeof(double));	
	
	int ranks[1], numprocs, myid, server, workerid;
	int request;
	double rands[nrand+1];

	if (argc==2)
		realization=atoi(argv[1]);
	else
		printf("One argument expected \n");	

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
		getMR(realization);						// get mock MR data (dithered from SLy values)
		readinSLY(p_SLY,epsilon_SLY, rho_SLY, &numlinesSLY);          	//Read in the low-density data from SLY
	}									// and broadcast to all other processes
	MPI_Bcast(m_data, nData+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Bcast(r_data, nData+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&moi_EOS, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sigma_moi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
	
	if (myid == server)
	{
		unsigned int Pseed;
		Pseed = mt_goodseed();

	}	
									
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
		sigmaPs = dvector(1, nparam);
		epspts = dvector(1, nparam);
		steps = dvector(1, nparam);
		gamma0_param = dvector(1,nparam-1);
		acoef_param = dvector(1,nparam-1);
		mm = dvector(1, nEpsilon);
		rr = dvector(1, nEpsilon);							

		p_ns = bisect_linint(rho_ns, rho_SLY, p_SLY,1, numlinesSLY);				//pressure at rho_saturation according to sly 
		getRhoPts(rhopts);								//Get the fiducial densities

		if (myid==0)
		{

			if (nparam==6)
			{
				initP(&mean_P1, &sigma_P1, &mean_P2, &sigma_P2, &mean_P3, &sigma_P3, &mean_P4, &sigma_P4, &mean_P5, &sigma_P5); 
				sigma_P1 /= 30.;
				sigma_P2 /= 30.;
				sigma_P3 /= 30.;
				sigma_P4 /= 30.;
				sigma_P5 /= 30.;
			}
			else
			{
				initP3(&mean_P1, &sigma_P1, &mean_P2, &sigma_P2, &mean_P3, &sigma_P3); 
				sigma_P1 /= 30.;
				sigma_P2 /= 30.;
				sigma_P3 /= 30.;
			}
		}

		MPI_Bcast(&mean_P1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mean_P2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mean_P3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		MPI_Bcast(&sigma_P1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&sigma_P2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&sigma_P3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		sigmaPs[1] = sigma_P1;
		sigmaPs[2] = sigma_P2;
		sigmaPs[3] = sigma_P3;

		if (nparam==6)
		{
			MPI_Bcast(&mean_P4, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&mean_P5, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&sigma_P4, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&sigma_P5, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			sigmaPs[4] = sigma_P4;
			sigmaPs[5] = sigma_P5;
		}

		int test_prior = 0;
		while (test_prior == 0)		//Get first set of starting Ps
		{
			MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);		//Send request for random numbers
			MPI_Recv(rands, nrand+1, MPI_DOUBLE, server, REPLY, world, &status); //Receive 7 random numbers
		
			Ppts_new[1] = p_ns;
			Ppts_new[2] = mean_P1*(rands[1]*3.);
			Ppts_new[3] = mean_P2*(rands[2]*3.);
			Ppts_new[4] = mean_P3*(rands[3]*3.);
			
			if (nparam==6)
			{
				Ppts_new[5] = mean_P4*(rands[4]*3.);
				Ppts_new[6] = mean_P5*(rands[5]*3.);
			}
	 		
			/*	
			Ppts_new[2] = bisect_linint(rhopts[2], rho_SLY, p_SLY,1, numlinesSLY); // 5.7748e33/p_char; // 				//pressure at rho_saturation according to sly 
			Ppts_new[3] = bisect_linint(rhopts[3], rho_SLY, p_SLY,1, numlinesSLY); //6.1207e34/p_char; // 				//pressure at rho_saturation according to sly 
			Ppts_new[4] = bisect_linint(rhopts[4], rho_SLY, p_SLY,1, numlinesSLY); //1.259e35/p_char; // 				//pressure at rho_saturation according to sly 
			Ppts_new[5] = bisect_linint(rhopts[5], rho_SLY, p_SLY,1, numlinesSLY); //4.4822e35/p_char; //				//pressure at rho_saturation according to sly 
			Ppts_new[6] = bisect_linint(rhopts[6], rho_SLY, p_SLY,1, numlinesSLY); //1.177e36/p_char; //				//pressure at rho_saturation according to sly
			

			Ppts_new[2] =5.831500e+33/p_char; // 				//"best-fit" pressure (from "most likely" EoS)
			Ppts_new[3] =9.589500e+34/p_char; // 				//"best-fit" pressure (from "most likely" EoS)
			Ppts_new[4] =9.644700e+34/p_char; // 				//"best-fit" pressure (from "most likely" EoS) 
			Ppts_new[5] =4.375400e+35/p_char; //				//"best-fit" pressure (from "most likely" EoS) 
			Ppts_new[6] =1.569700e+36/p_char; //				//"best-fit" pressure (from "most likely" EoS) 
			*/

			test_prior=1;

			if (Ppts_new[2] < 1.35195e-5 || Ppts_new[3] < 4.42759e-5)	// nn lower limit on P1 and P2:  P1 >= 3.60 MeV/fm^3, P2 >= 11.79 MeV/fm^3
				test_prior = 0;			
			else
			{
				edgeConditions(&r_start, &Pedge);
				initConditions(initM, initRho, initEpsilon, centralP, Ppts_new, gamma0_param, acoef_param, epspts);
				for (j=2; j<=nparam; j++)
				{
					test_prior *= prior_P(j,Ppts_new,epspts, gamma0_param,acoef_param);
					if (test_prior==0)
						continue;
				}

				tovI(Ppts_new, rr, mm, gamma0_param, acoef_param,inertia);			  //Get mass, radius, and update gamma and 'a' coefs for first set of Ppts
				maxM = max_array(mm,nEpsilon);	

				if (maxM < 1.97)
					test_prior=0;
			}
		}


		edgeConditions(&r_start, &Pedge);
		initConditions(initM, initRho, initEpsilon, centralP, Ppts_new, gamma0_param, acoef_param, epspts);

			
		char filename[260];
		sprintf(filename,"chain_output_SLY_%d_I/inversion_output_%d.txt",realization, myid);
		//sprintf(filename,"test/inversion_usingBestPs_SLY_5_%d.txt",myid);
		FILE *f = fopen(filename,"w");
		if (nparam==6)
		{
			fprintf(f,"File created by inversion.c, assuming a 5-polytrope parametrized EoS throughout the star (P0 = P_sat) \n");
			fprintf(f,"Starting pressures from histogram of 42 EoS and dithered by: %f %f %f %f %f\n", rands[1]*3., rands[2]*3., rands[3]*3., rands[4]*3., rands[5]*3.);
			fprintf(f,"N_MC and Acceptance rate details in footer. \n");
			fprintf(f,"Posterior      P_1         P_2          P_3        P_4          P_5 \n");
		}
		else
		{
			fprintf(f,"File created by inversion.c, assuming a 3-polytrope parametrized EoS throughout the star (P0 = P_sat) \n");
			fprintf(f,"Starting pressures from histogram of 42 EoS and dithered by: %f %f %f \n", rands[1]*3., rands[2]*3., rands[3]*3.);
			fprintf(f,"N_MC and Acceptance rate details in footer. \n");
			fprintf(f,"Posterior      P_1         P_2          P_3    \n");
		}

		fflush(f);

		/*
		tovI(Ppts_new, rr, mm, gamma0_param, acoef_param, inertia);			  //Get mass, radius, and update gamma and 'a' coefs for first set of Ppts
		posterior_old = getPosterior(Ppts_new, epspts, gamma0_param,acoef_param,		  //Get posterior likelihood for original set of Ppts
 					     mm, m_data, m_sigma,rr, r_data, r_sigma);
		
		
		fprintf(f, "%e %e %e %e %e %e\n", posterior_old, Ppts_new[2]*p_char, Ppts_new[3]*p_char, Ppts_new[4]*p_char, Ppts_new[5]*p_char, Ppts_new[6]*p_char);

		for (i=1; i<=nEpsilon;i++)
			fprintf(f,"%f %f %e\n", rr[i], mm[i], inertia[i]);
		fflush(f);
		exit(0);
		*/

		double M_low = 0.2;
		double M_up = 3.1;
		double M_delta;
		double R_low = 8;
		double R_up = 15;
		double R_delta;
		
		M_delta = (M_up - M_low)/nMRgrid;
		R_delta = (R_up - R_low)/nMRgrid;

		int **hist;

		Mgrid = (double*)malloc((nMRgrid+1)*sizeof(double));
		Rgrid = (double*)malloc((nMRgrid+1)*sizeof(double));
		hist = imatrix(1, nMRgrid-1, 1, nMRgrid-1);	

		for (i=1; i<=nMRgrid-1; i++)
		{
			for (j=1; j<=nMRgrid-1; j++) hist[i][j]=0;
		}

		Mgrid[1] = M_low;
		Rgrid[1] = R_low;
		for (j=2; j<=nMRgrid; j++)
		{
			Mgrid[j] = Mgrid[j-1] + M_delta;
			Rgrid[j] = Rgrid[j-1] + R_delta;
		}

	
		/*
		getMRhist(myid, mm, rr, hist);
	
		char name2[260];
		sprintf(name2,"margMR_output_sly/MRgrid_%d.txt",myid);
		FILE *fgrid = fopen(name2,"w");
		fprintf(fgrid, "File created by inversion.c \n");
		fprintf(fgrid, "%d grid points:  M grid from %f-%f, R grid from %f-%f\n",nMRgrid, M_low, M_up, R_low, R_up);
		fflush(fgrid);
		*/

		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);		//Send request for random numbers

		i=1;
		nMC_thisrank = nMC/(numprocs-1);
		while (i <= nMC_thisrank)						   //Loop through all MC points
		{
			MPI_Recv(rands, nrand+1, MPI_DOUBLE, server, REPLY, world, &status); //Receive 3 random numbers for this loop iteration

			if (i != 1 && laststep_rejected == 0)				   //If we moved to a new step (rejected=0), use that posterior as the starting posterior
				posterior_old = posterior_new;				   // (otherwise, posterior_old is still posterior_old, so we don't have to update it)

			toContinue=0;
			gaussian(steps, 0.,sigmaPs, rands);
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

			initConditions(initM, initRho, initEpsilon, centralP, Ppts_new, gamma0_param, acoef_param, epspts); //Get new init values for new set of Ppts
			
			tovI(Ppts_new,rr, mm, gamma0_param,acoef_param,inertia);			   //Get M,R and update gamma and a_coef for NEW set of Ppts

			posterior_new = getPosterior(Ppts_new, epspts, gamma0_param,acoef_param,   //Get posterior likelihood for NEW set of Ppts
						    mm, m_data, m_sigma,rr, r_data, r_sigma, inertia);
			ratio = posterior_new / posterior_old;
			scriptP = rands[7];						   //MT pseudorandom double in [0,1) with 32 bits of randomness 
			
			if (scriptP < ratio)						   //If new P points are accepted
			{
				laststep_rejected = 0;
				
				if (nparam==6)
					fprintf(f, "%e  %6.4e  %6.4e  %6.4e  %6.4e   %6.4e \n",posterior_new, Ppts_new[2]*p_char, Ppts_new[3]*p_char, Ppts_new[4]*p_char,Ppts_new[5]*p_char,Ppts_new[6]*p_char);
				else
					fprintf(f, "%e  %6.4e  %6.4e  %6.4e \n",posterior_new, Ppts_new[2]*p_char, Ppts_new[3]*p_char, Ppts_new[4]*p_char);
				fflush(f);
				accepted++;

				//maxM = max_array(mm,nEpsilon);	
				//firstMax = mm [ firstmax(mm,nEpsilon) ];
				//getMRhist(myid, mm, rr, hist);

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
			
			if (request)
				MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);

		}

		fprintf(f,"Total steps attempted in this process: %d nBurnIn(number accepted that are thrown out): %d \n", attempted, nBurn);
		fprintf(f,"Total steps attempted after BurnIN: %d  Number accepted after BurnIn: %d \n", attempted-nBurn, accepted);
		fclose(f);	

		/*
		fprintf(fgrid, "R(centered)   M(centered)      Count \n");
		for (k=1; k<=nMRgrid-1; k++)
		{
			for (l=1; l<=nMRgrid-1; l++)  fprintf(fgrid, "%f   %f   %ld \n",(Mgrid[k]+Mgrid[k+1])/2., (Rgrid[l]+Rgrid[l+1])/2., hist[k][l]); 
			fflush(fgrid);
		}
		fclose(fgrid);
		*/

		free_dvector(epspts,1,nparam);
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
		free_dvector(inertia, 1, nEpsilon);
	
		free_imatrix(hist,1,nMRgrid-1, 1, nMRgrid-1);		
		free(Mgrid);
		free(Rgrid);

	}

	if (myid != server )
		MPI_Barrier(workers);
	
	if (myid ==0)
	{
		request=0;
		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
	}

	if (myid != server)
		MPI_Comm_free(&workers);


	free_dvector(m_data,1,nData+1);
	free_dvector(r_data,1,nData+1);
	free_dvector(m_sigma,1,nData+1);
	free_dvector(r_sigma,1,nData+1);
	MPI_Finalize();
	return 0;
}

void getMRhist(int myid, double *mm, double *rr, int **hist)
{

	int i,j,k, max1_index;
	int nTot, nExtra = 10;	//Number of MR points to linearly interpolate between each set of MR pts
	double *m_hr, *r_hr;	//Mass and Radius arrays for our high-res (interpolated) curve
	double slope, b, deltaR;

	nTot = (nEpsilon-1)*nExtra + nEpsilon;	//Total number of MR points in new, high-res curve

	m_hr = (double*)malloc((nTot+1)*sizeof(double));
	r_hr = (double*)malloc((nTot+1)*sizeof(double));
	
	k=1;
	for (i=1; i<=nEpsilon-1; i++)			//Linearly interpolate between each set of MR points
	{
		slope = (mm[i]-mm[i+1])/(rr[i]-rr[i+1]);
		b = mm[i] - slope*rr[i];
		deltaR = (rr[i+1] - rr[i])/(nExtra+1.);	//Spacing between nExtra linearly interpolated points

		m_hr[k] = mm[i];
		r_hr[k] = rr[i];				
		k++;

		for (j=1; j<=nExtra; j++)
		{
			r_hr[k] = r_hr[k-1] + deltaR;
			m_hr[k] = slope*r_hr[k] + b;		
			k++;
		}
	}
	m_hr[k] = mm[nEpsilon];	//don't miss last point!
	r_hr[k] = rr[nEpsilon]; //don't miss last point!
	
	max1_index = firstmax(m_hr, nTot);

	/* Now count which squares of the MR grid this high-res curve passes through */

	for (i=1; i<=nMRgrid-1; i++)		//Loop through M-dimension of grid
	{
		for (j=1; j<=nMRgrid-1; j++)	//Loop through R-dimension of grid
		{

			for (k=1; k<=max1_index; k++)	//Loop through high-res MR curve
			{
				if ( m_hr[k] >= Mgrid[i] && m_hr[k] < Mgrid[i+1] && r_hr[k] >= Rgrid[j] && r_hr[k] < Rgrid[j+1])
				{	
					hist[i][j] += 1;			
					break; 	//once we increase a grid of our histogram, move on (i.e., don't double count)
				}
			}
		}
	}

	free(m_hr);
	free(r_hr);
}

int firstmax(double *array, int length)
{
	int i, index;
	
	for (i=1; i<=length-3; i++)
	{
		//if (array[i] < 1. || array[i] <= array[i+1] || array[i+1] <= array[i+2] || array[i+2] <= array[i+3])
		if (array[i] < 0.2 || array[i] <= array[i+1] )
			index=i+1;	//If the next three M pts are higher, (and we're above M=1), not yet at the max
		else
			break;		//max = where next three M pts are lower (and above M=1)

	}
	return index;
}

void revertP(double Ppts_new[], double Ppts_old[])
/* Go back to the old values of Ppts if a new step was rejected */
{
	int j;
	for (j=2; j<=nparam; j++)
		Ppts_new[j] = Ppts_old[j];
}


double getPosterior(double Ppts_local[], double epspts_local[],  double gamma0_param[], double acoef_param[], double mm[], double m_data[], double m_sigma[], double rr[], double r_data[], double r_sigma[], double inertia[])
/* Get the posterior likelihood for a set of Ppts using
 	P(P1,...,P5 | data) ~ P(data | P1,...,P5) * [P_pr(P1)* ... *P_pr(P5)] 	  
 */
{
	int j,k,index, maxM_index, max1_index, max1_index_grid;
	int ngrid = 200;
	int stable_resumes, twins;
	double m_min;
	double likelihood, posterior, maxM, firstMax;
	double *unMarg_L;
	double *r_j, *m_j;
	double L_moi, moi_model;
	r_j = (double*)malloc((ngrid+1)*sizeof(double));
	m_j = (double*)malloc((ngrid+1)*sizeof(double));
	unMarg_L = (double*)malloc((ngrid+1)*sizeof(double));
	maxM = max_array(mm,nEpsilon);				//the maximum mass achieved for this set of P1, ..., P5
	maxM_index = max_array_index(mm, nEpsilon);

	double deltaM, slope, intercept;
	m_min = 0.1;  // mm[minData_index]; 

	max1_index = firstmax(mm,nEpsilon);
	firstMax = mm[max1_index];

	if (maxM-firstMax > 0.1)					//If there is more than 1 local maximum in mass
	{
		twins = 1;					//Set flag indicating multiple stable branches (twins)
		j=max1_index;
		while ( mm[j+1] < mm[j] )//&& mm[j+2] < mm[j+1] )	//While we're still on the decreasing (unstable) branch
			j++;					// increase counter until we reach the stable branch
		
		stable_resumes = j;
	
		deltaM = (maxM - mm[stable_resumes] + mm[max1_index] - m_min)/ngrid; 

		m_j[1] = m_min;
		j=1;
		while (m_j[j] <= mm[max1_index])			//Get high-res grid up to the first maximum
		{
			r_j[j] = bisect_linint(m_j[j], mm, rr, 1, max1_index);
			m_j[j+1] = m_j[j] + deltaM;
			j+=1;
		}
		max1_index_grid = j-1;
		m_j[max1_index_grid+1] = mm[stable_resumes]; 
		for (j=max1_index_grid+1; j<=ngrid; j++)		//Get the high-res grid for the 2nd stable branch
		{
			r_j[j] = bisect_linint(m_j[j], mm, rr, stable_resumes, maxM_index);
			if (j<ngrid) m_j[j+1] = m_j[j] + deltaM;
		}
	} else
	{
		twins = 0;					//Otherwise, set flag indicating only 1 maximum

		deltaM = (maxM - m_min)/ngrid;
		max1_index_grid = 0;
		m_j[1] = m_min;
		for (j=1; j<=ngrid; j++)					//Get the 200-pt resolution grid in M,R for this EOS
		{
			r_j[j] = bisect_linint(m_j[j], mm, rr, 1, maxM_index);
			if (j<ngrid) m_j[j+1] = m_j[j] + deltaM;
		}
	}

	likelihood = 1.0;					//P(data | P1, ..., P5)
	for (k=1; k<=nData; k++)
	{
		for (j=1; j<=ngrid; j++)
			unMarg_L[j] = getLikelihood(m_sigma[k], r_sigma[k], m_data[k], r_data[k], m_j[j], r_j[j]);
		likelihood *= integrateM(unMarg_L, max1_index_grid, ngrid, deltaM,  m_j, r_j, r_sigma[k], m_sigma[k]);
	}

	moi_model = bisect_linint(1.338, mm, inertia, 1, nEpsilon);
	L_moi = exp(-(moi_EOS - moi_model)*(moi_EOS - moi_model) / (2.*sigma_moi*sigma_moi) );

	posterior = likelihood*L_moi;
	
	if (Ppts_local[2] <=1.35195e-5  || Ppts_local[3] <= 4.42759e-5 || maxM < 1.97)	// Prior on P1 and P2 from nn-interaction:  P1 >= 3.60MeV/fm^3, P2 >= 11.79MeV/fm^3
		posterior = 0.;					
	else
	{
		for (j=2; j<=nparam; j++)
			posterior *= prior_P(j,Ppts_local,epspts_local, gamma0_param,acoef_param);			//Multiply the posterior likelihood by the priors
	}

	free(unMarg_L);
	free(r_j);
	free(m_j);

	return posterior;
}

double getLikelihood(double sigmaM, double sigmaR, double dataM, double dataR, double EOSm, double EOSr)
{
	double L;
	L = (1./(2.*M_PI*sigmaM*sigmaR))*exp(-(dataM - EOSm)*(dataM - EOSm)/(2.*sigmaM*sigmaM) - 
			(dataR - EOSr)*(dataR - EOSr)/(2.*sigmaR*sigmaR) );
	return L;
}

double prior_P(int j, double Ppts_local[], double epspts_local[], double gamma0_param[], double acoef_param[])
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
	double gamfactor = 3.;	//Factor by which adjacent (non-PT) values of gamma must agree
	double oneOver = 0.3333; // =1/gamfactor

	/*  Priors: P_i > P_(i-1), the gamma leading up to P_i must not be acausal, adj. (non-PT) gammas must not differ by more than a factor of 5 */
	//if (j < nparam && gamma0_param[j-1] > 0.05 && (gamma0_param[j-1] > gamfactor*gamma0_param[j] || gamma0_param[j-1] < oneOver*gamma0_param[j]) ) 
	//	prior=0.;
	//else
	//{

	if (Ppts_local[j] < Ppts_local[j-1] || getCs(j, Ppts_local, epspts_local) > 1.)
		prior=0.;
	//
	//}
	return prior;
}

	
double integrateM(double unMarg_L[],int unstable1, int unstable2, double deltaM, double mgrid[], double rgrid[], double sigmaR, double sigmaM)
/*
 For the M-R curve from a given set of (P1, ..., P5), marginalize over M_rhoc:

 P(M, R | P1,... P5) = C*Integral_Mmin^Mmax P(M, R(M) | P1,... P5) P_pr(M) dM
 
*/ 
{	
	int j, index;
	double likelihood = 0.;
	double prior_m = 1.0;
	double jeffreys, dRdM;
	double sigR2 = 1./(sigmaR*sigmaR);
	double sigM2 = 1./(sigmaM*sigmaM);
	
	if (unstable1 > 0)
	{
		for (j=2; j<=unstable1; j++)
		{
			dRdM = (rgrid[j-1]-rgrid[j]) / (mgrid[j-1]-mgrid[j]);		
			jeffreys = sqrt( sigM2 + dRdM*dRdM*sigR2 );
			likelihood += unMarg_L[j]*jeffreys*deltaM;
		}
	}
	for (j=unstable1+2; j<=unstable2; j++)
	{
		dRdM = (rgrid[j-1]-rgrid[j]) / (mgrid[j-1]-mgrid[j]);		
		jeffreys = sqrt( sigM2 + dRdM*dRdM*sigR2 );
		likelihood += unMarg_L[j]*jeffreys*deltaM;
	}

	return likelihood;
}

	
double integrateRhoC(double unMarg_L[], int maxM_index)
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
	double deltaRho = (initRho[maxM_index]-initRho[1])/(ngrid);  	//Range of integration is from rhoC_min to rhoC_(MR curve turnover)
	double m_i, rho_i, m, intercept, unMarg_L_i;
	rho_i = initRho[1];

	for (j=1; j<=ngrid+1; j++)
	{
		index = bisection(rho_i, initRho, maxM_index);		//Find the nearest value of rho_c (w/in the STABLE branch) that was actually calculated

		if (initRho[index] > rho_i)	//If the point we found is above rho_i
		{
			m = (unMarg_L[index] - unMarg_L[index-1]) / 	//Calculate the slope between it and the rho below rho_i 
		    	    (initRho[index] - initRho[index-1]);	
		}
		else							//Otherwise, if it is below rho_i
		{
			m = (unMarg_L[index] - unMarg_L[index+1]) /	//Calculate the slope between it and the rho above rho_i 
		    	    (initRho[index] - initRho[index+1]);
		}
	
		intercept = unMarg_L[index] - m*initRho[index];		//Calculate y-intercept
		unMarg_L_i = m*rho_i + intercept;			//Find linear interpolation of the likelihood
	
		likelihood += unMarg_L_i*prior_rho*deltaRho;
		rho_i += deltaRho;
	}
	return likelihood;
}

double getCs(int i, double Ppts_local[], double epspts_local[] )
{
	double cs;

	if (epspts_local[i] == epspts_local[i-1])
		cs=0.;
	else
		cs = sqrt( (Ppts_local[i] - Ppts_local[i-1]) / (epspts_local[i] - epspts_local[i-1]) );

	return cs;
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


void gaussian(double *steps, double mean, double *sigmas, double *rands)
/* Draw a random number from a Gaussian distribution */
{
	steps[1] = sqrt(-2.*log(rands[1]))*cos(2.*M_PI*rands[2])*sigmas[1] + mean;
	steps[2] = sqrt(-2.*log(rands[1]))*sin(2.*M_PI*rands[2])*sigmas[2] + mean;

	steps[3] = sqrt(-2.*log(rands[3]))*cos(2.*M_PI*rands[4])*sigmas[3] + mean;

	if (nparam==6)
	{
		steps[4] = sqrt(-2.*log(rands[3]))*sin(2.*M_PI*rands[4])*sigmas[4] + mean;
		steps[5] = sqrt(-2.*log(rands[5]))*cos(2.*M_PI*rands[6])*sigmas[5] + mean;
	}
}

