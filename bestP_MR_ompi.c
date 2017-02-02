/* C. Raithel, May 2016
  
   This program finds the best fit values of (P1, ..., P5) from the output
   of inversion.c by searching the (5D) posterior likelihoods for all values
   within 1/e of the max value.
   
   It then runs the parametric version of the TOV solver to get the M,R curve 
   corresponding to each set of most likely (P2, ..., P5) and outputs these allowed
   M,R points to a single file.
 
   To compile:
   gcc bestP_MR.c ../nrutil.c useful_funcs.c tov_helper.c tov_polyparam.c -o bestP_MR -lm

  */

#include "../nrutil.h"
#include "header_useful.h"
#include "mtwist-1.5/mtwist.h"
#include "mpi.h"
#include "tov_dev.h"
#include "bestPMR.h"
#include <omp.h>

#define cutoff "1overE" //"80pct"   	 //What fraction within L_max to include
#define ext "_designer2_0_uniPr_reg4_3param_kmerr"
#define test_ext "5e5"
#define nSampled 1280 //1280 //5120  //10240	 //Number of 

#define nFiles 369 //119
#define nMCperfile 5027 // 15500 //6400 //2711//8130 // 13550 //18970 // 25600
#define nMC 1860000 //2160000 //2368000 //1003070 // 3008100 // 5013500 // 7018900 // 9472000  //(nFiles+1)*nMCperfile
#define nrand 1
#define REQUEST 1
#define REPLY 2

void readin5seg(double *post_host, double *P1_host, double *P2_host, double *P3_host, double *P4_host, double *P5_host);
void readin3seg(double *post_host, double *P1_host, double *P2_host, double *P3_host);

int main( int argc, char *argv[])
{
	int i,j, k, numprocs, myid, nSLY_host, max_index;
	double *post_host, *P1_host, *P2_host, *P3_host, *P4_host, *P5_host;
	double *post_1overE, *P1_1overE, *P2_1overE, *P3_1overE, *P4_1overE, *P5_1overE; 
	double *post_1overE_tot, *P1_1overE_tot, *P2_1overE_tot, *P3_1overE_tot, *P4_1overE_tot, *P5_1overE_tot; 
	double *rho_SLY_host, *p_SLY_host, *eps_SLY_host;
	double max_P_host, oneOverE;


	int required=MPI_THREAD_FUNNELED;
	int provided;
	int ranks[1], server, workerid;
	int request, rand_int;
	double rands[nrand+1];
	MPI_Comm world, workers;
	MPI_Group world_group, worker_group;
	MPI_Status status;
	
	MPI_Init_thread(&argc, &argv, required, &provided);
	world = MPI_COMM_WORLD;
	MPI_Comm_size(world, &numprocs);
	MPI_Comm_rank(world, &myid);
	server = numprocs-1;				 			//last proc is server for random numbers

	if (provided < required) 		//if threading support level is not available
	{
		printf("WARNING: Insufficient threading support. Forcing nthread=1. \n");
		omp_set_num_threads(1);
	}

	p_SLY_host = (double*)malloc((lines+1)*sizeof(double));
	rho_SLY_host = (double*)malloc((lines+1)*sizeof(double));
	eps_SLY_host = (double*)malloc((lines+1)*sizeof(double));

	post_host = (double*)malloc((nMC+1)*sizeof(double));
	P1_host = (double*)malloc((nMC+1)*sizeof(double));
	P2_host = (double*)malloc((nMC+1)*sizeof(double));
	P3_host = (double*)malloc((nMC+1)*sizeof(double));

	if (nparam==6)
	{
		P4_host = (double*)malloc((nMC+1)*sizeof(double));
		P5_host = (double*)malloc((nMC+1)*sizeof(double));
	}


	if (myid == 0)
	{
		if (nparam==6)
			readin5seg(post_host, P1_host, P2_host, P3_host, P4_host, P5_host);
		else
			readin3seg(post_host, P1_host, P2_host, P3_host);
		readinSLY(p_SLY_host, eps_SLY_host, rho_SLY_host, &nSLY_host);

	}

	MPI_Bcast(post_host, nMC+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P1_host, nMC+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P2_host, nMC+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P3_host, nMC+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (nparam==6)
	{
		MPI_Bcast(P4_host, nMC+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(P5_host, nMC+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	MPI_Bcast(p_SLY_host, lines, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(eps_SLY_host, lines, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(rho_SLY_host, lines, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nSLY_host, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	max_P_host = max_array(post_host,nMC);
	max_index = max_array_index(post_host, nMC);

	if (strcmp(cutoff,"1overE")==0)
		oneOverE = max_P_host * exp(-1.);
	else if (strcmp(cutoff, "80pct")==0)
		oneOverE = max_P_host*0.8;
	else if (strcmp(cutoff, "68pct")==0)
		oneOverE = max_P_host*0.68;


	int nMC_thisrank, start, stop;
	nMC_thisrank = nMC/numprocs;

	int n1overE_tot = 0;
	int n1overE, nPerProc;
	for (i=1; i<=nMC; i++)
		if (post_host[i] >= oneOverE) n1overE_tot++;

	nPerProc = nSampled/(numprocs-1);
	double p_ns = bisect_linint(rho_ns, rho_SLY_host, p_SLY_host,1, nSLY_host);				//pressure at rho_saturation according to sly 

	if (nSampled > n1overE_tot)
	{
		printf("nSampled %d > n1overE %d \n *** PROGRAM TERMINATING *** \n", nSampled, n1overE_tot);
		fflush(stdout);

		if (myid==0 & nparam==6)
		{
			FILE *fbest = fopen("Ps_best.txt", "w");
			fprintf(fbest, "highest posterior: %e \n", max_P_host);
			fprintf(fbest, "%e   %e \n", 2.7e14, p_ns*p_char);
			fprintf(fbest, "%e   %e \n", 3.78e14, P1_host[max_index]*p_char);
			fprintf(fbest, "%e   %e \n", 5.94e14, P2_host[max_index]*p_char);
			fprintf(fbest, "%e   %e \n", 8.91e14, P3_host[max_index]*p_char);
			fprintf(fbest, "%e   %e \n", 13.23e14, P4_host[max_index]*p_char);
			fprintf(fbest, "%e   %e \n", 19.98e14, P5_host[max_index]*p_char);
			fclose(fbest);
		}

		exit(0);
	}

	post_1overE_tot = (double*)malloc((n1overE_tot+1)*sizeof(double));
	P1_1overE_tot = (double*)malloc((n1overE_tot+1)*sizeof(double));
	P2_1overE_tot = (double*)malloc((n1overE_tot+1)*sizeof(double));
	P3_1overE_tot = (double*)malloc((n1overE_tot+1)*sizeof(double));
	post_1overE = (double*)malloc((nPerProc+1)*sizeof(double));
	P1_1overE = (double*)malloc((nPerProc+1)*sizeof(double));
	P2_1overE = (double*)malloc((nPerProc+1)*sizeof(double));
	P3_1overE = (double*)malloc((nPerProc+1)*sizeof(double));

	if (nparam==6)
	{
		P4_1overE_tot = (double*)malloc((n1overE_tot+1)*sizeof(double));
		P5_1overE_tot = (double*)malloc((n1overE_tot+1)*sizeof(double));
		P4_1overE = (double*)malloc((nPerProc+1)*sizeof(double));
		P5_1overE = (double*)malloc((nPerProc+1)*sizeof(double));
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

		j=1;
		for (i=1; i<=nMC; i++)			//Find all P's with likelihood >= 1/E max
		{
			if (post_host[i] >= oneOverE)
			{
				post_1overE_tot[j] = post_host[i];
				P1_1overE_tot[j] = P1_host[i];
				P2_1overE_tot[j] = P2_host[i];
				P3_1overE_tot[j] = P3_host[i];
				if (nparam==6)
				{
					P4_1overE_tot[j] = P4_host[i];
					P5_1overE_tot[j] = P5_host[i];
				}
				j++;
			}
		}

		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
		j=1;
		while (j<=nPerProc)
		{

			MPI_Recv(rands, nrand+1, MPI_DOUBLE, server, REPLY, world, &status); 	//Receive 2 random numbers
			rand_int = 1 + floor( rands[1]*n1overE_tot);

			if (myid==0 && j==1)
			{
				post_1overE[j] = post_host[max_index];
				P1_1overE[j] = P1_host[max_index];    //Force the maximum likelihood to be included		
				P2_1overE[j] = P2_host[max_index];
				P3_1overE[j] = P3_host[max_index];
	
				if (nparam==6)
				{
					P4_1overE[j] = P4_host[max_index];
					P5_1overE[j] = P5_host[max_index];
				}

			} else
			{
				post_1overE[j] = post_1overE_tot[rand_int];
				P1_1overE[j] = P1_1overE_tot[rand_int];
				P2_1overE[j] = P2_1overE_tot[rand_int];
				P3_1overE[j] = P3_1overE_tot[rand_int];

				if (nparam==6)
				{
					P4_1overE[j] = P4_1overE_tot[rand_int];
					P5_1overE[j] = P5_1overE_tot[rand_int];
				}
			}
			j++;	
			
			if (j <= nPerProc)
				MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
			

		}

		char filename[256];
		char MRfile[256];

		if (strcmp(cutoff,"80pct")==0)
		{
			sprintf(filename, "MR_output%s_80pct/Ps_%d.txt",ext,  myid);
			sprintf(MRfile, "MR_output%s_80pct/MR_%d.txt",ext, myid);
		}
		else
		{
			sprintf(filename, "MR_output%s/Ps_%d.txt",ext, myid);
			sprintf(MRfile, "MR_output%s/MR_%d.txt",ext, myid);
		}

		FILE *f_P = fopen(filename, "w");
		fprintf(f_P, "Randomly sampled sets of (P1,...,P5) drawn from the set of P's with 5-dimensional likelihoods within %s OF MAX (%d total over threshold) \n",cutoff, n1overE_tot); // 1/E of the maximum\n");
		fflush(f_P);

		
		if (nparam==6)	
		{
			for (j=1; j<=nPerProc; j++)
				fprintf(f_P, "%e  %e %e %e %e %e %e \n",post_1overE[j],  p_ns*p_char, P1_1overE[j]*p_char,P2_1overE[j]*p_char,P3_1overE[j]*p_char,P4_1overE[j]*p_char,P5_1overE[j]*p_char);
		} else
		{
			for (j=1; j<=nPerProc; j++)
				fprintf(f_P, "%e  %e %e %e %e \n",post_1overE[j],  p_ns*p_char, P1_1overE[j]*p_char,P2_1overE[j]*p_char,P3_1overE[j]*p_char);
		}
		fclose(f_P);
	
		if (nparam==6)
			omp_driver(MRfile, myid,  nPerProc, rho_SLY_host, p_SLY_host, eps_SLY_host, nSLY_host, nparam-1, P1_1overE, P2_1overE, P3_1overE, P4_1overE, P5_1overE);	
		else
			omp_driver(MRfile, myid,  nPerProc, rho_SLY_host, p_SLY_host, eps_SLY_host, nSLY_host, nparam-1, P1_1overE, P2_1overE, P3_1overE);	


	}


	if (myid==0)
	{
		request=0;
		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
	}

	if (myid != server)
		MPI_Comm_free(&workers);


	free(post_host);
	free(P1_host);
	free(P2_host);
	free(P3_host);
	free(rho_SLY_host);
	free(p_SLY_host);
	free(eps_SLY_host);
	free(post_1overE);
	free(P1_1overE);
	free(P2_1overE);
	free(P3_1overE);
	free(post_1overE_tot);
	free(P1_1overE_tot);
	free(P2_1overE_tot);
	free(P3_1overE_tot);

	if (nparam==6)
	{
		free(P4_host);
		free(P5_host);
		free(P4_1overE);
		free(P5_1overE);
		free(P4_1overE_tot);
		free(P5_1overE_tot);
	}

	MPI_Finalize();
	return 0;
}

void readin5seg(double *post_host, double *P1_host, double *P2_host, double *P3_host, double *P4_host, double *P5_host)
/*  Format to read in MCMC chain output for 5-segment param EoS */
{
	int i, j, k;
	char infile[256], dir[256];
	char buff[256];
	FILE *file;

	sprintf(dir, "chain_output%s",ext);	
	i=1;
	for (j=0; j <= nFiles; j++)
	{
		sprintf(infile,"%s/inversion_output_%d.txt",dir,j);
		file = fopen(infile, "rt");
		fgets(buff, 256, file);
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	

		for (k=1; k<=nMCperfile; k++)
		{
			if  (fscanf(file, "%le %le %le %le %le %le", &post_host[i], &P1_host[i], &P2_host[i], &P3_host[i], &P4_host[i], &P5_host[i]) == 6)
			{
				P1_host[i] /= p_char;
				P2_host[i] /= p_char;
				P3_host[i] /= p_char;
				P4_host[i] /= p_char;
				P5_host[i] /= p_char;
				i++;
			}
		}
	}
}

void readin3seg(double *post_host, double *P1_host, double *P2_host, double *P3_host)
/*  Format to read in MCMC chain output for 3-segment param EoS */
{
	int i, j, k;
	char infile[256], dir[256];
	char buff[256];
	FILE *file;
	
	sprintf(dir, "chain_output%s",ext);	
	i=1;
	for (j=0; j <= nFiles; j++)
	{
		sprintf(infile,"%s/inversion_output_%d.txt",dir,j);
		file = fopen(infile, "rt");

		fgets(buff, 256, file);
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	
		fgets(buff, 256, file);	
		for (k=1; k<=nMCperfile; k++)
		{
			if  (fscanf(file, "%le %le %le %le ", &post_host[i], &P1_host[i], &P2_host[i], &P3_host[i]) == 4)
			{
				P1_host[i] /= p_char;
				P2_host[i] /= p_char;
				P3_host[i] /= p_char;
				i++;
			}
		}
	}
}
