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
#include "tov.h"
#include "mpi.h"

#define nMC 36800
#define nMCperfile 100

double *rhopts; 							//-------------------------------------------------------//  
double *p_SLY, *epsilon_SLY, *rho_SLY;					// Re-initialize GLOBAL variables defined in header file //
double *initM, *initRho, *initEpsilon, *centralP;			//							 //
double r_start, Pedge, p_ns;						//							 //
int numlinesSLY;							//-------------------------------------------------------//

int main(int argc, char *argv[])
{
	int myid, numprocs;
	MPI_Init( &argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	int i,j;
	int nMC_thisrank, nMC_start, nMC_end;
	double *post, *P1, *P2, *P3, *P4, *P5;
	double *Ppts, *gamma0_param, *acoef_param;
	double *rr, *mm;
	double max_P, oneOverE, oneOverSqrtE;
	char buff[256];
	FILE *file;

	initRho = dvector(1, nEpsilon);
	initEpsilon = dvector(1, nEpsilon);
	initM = dvector(1, nEpsilon);
	centralP = dvector(1, nEpsilon);

	rhopts = dvector(1, nparam);
	p_SLY = dvector(1, lines);
	rho_SLY = dvector(1, lines);
	epsilon_SLY = dvector(1, lines);

	post = dvector(1,nMC);
	P1 = dvector(1,nMC);
	P2 = dvector(1,nMC);
	P3 = dvector(1,nMC);
	P4 = dvector(1,nMC);
	P5 = dvector(1,nMC);
	Ppts = dvector(1,nparam);
	gamma0_param = dvector(1,nparam-1);
	acoef_param = dvector(1, nparam-1);
	rr = dvector(1, nEpsilon);
	mm = dvector(1, nEpsilon);

	if (myid == 0)
	{
		char infile[256];
		i=1;
		for (j=0; j<=368; j++)
		{
			sprintf(infile,"chain_output/inversion_output_%d.txt",j);
			file = fopen(infile, "rt");
			fgets(buff, 256, file);
			fgets(buff, 256, file);	
			fgets(buff, 256, file);	

			for (k=1; k<=nMCperfile; i++)
			{
				fscanf(file, "%le %le %le %le %le %le", &post[i], &P1[i], &P2[i], &P3[i], &P4[i], &P5[i]);
				P1[i] /= p_char;
				P2[i] /= p_char;
				P3[i] /= p_char;
				P4[i] /= p_char;
				P5[i] /= p_char;
				i++;
			}
			fclose(file);
		}
	}
	MPI_Bcast(post, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P1, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P2, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P3, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P4, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(P5, nMC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	max_P = max_array(post,nMC);
	oneOverE = max_P * exp(-1.);
	oneOverSqrtE = max_P * exp(-0.5);
	
	edgeConditions(&r_start, &Pedge);
	readinSLY(p_SLY, epsilon_SLY, rho_SLY, &numlinesSLY);
	getRhoPts(rhopts);
	
	p_ns = bisect_linint(rho_ns, rho_SLY, p_SLY, numlinesSLY);				

	char filename1[260];
	sprintf(filename1, "MR_output/MR_oneOverE_%d.txt",myid);
	FILE *f_oneOverE = fopen(filename1,"w");

	char filename2[260];
	sprintf(filename2, "MR_output/MR_oneOverSqrtE_%d.txt",myid);
	FILE *f_oneOverSqrtE = fopen(filename2,"w");

	nMC_thisrank = nMC/numprocs;			//Num MC points to loop thru in this process
	nMC_start = nMC_thisrank*(myid+1) - (nMC_thisrank -1);  //Starting index (wrt ALL MC points)
	nMC_end = nMC_thisrank*(myid+1);
	
	for (i=nMC_start; i<=nMC_end;i++)
	{
	
		if (post[i] >= oneOverE)
		{
			Ppts[1] = p_ns;
			Ppts[2] = P1[i];
			Ppts[3] = P2[i];
			Ppts[4] = P3[i];
			Ppts[5] = P4[i];
			Ppts[6] = P5[i];

			initConditions(initM, initRho, initEpsilon, centralP, Ppts, gamma0_param, acoef_param);
			tov(Ppts, rr, mm,gamma0_param, acoef_param);

			for (j=1; j<=nEpsilon; j++) fprintf(f_oneOverE, "%f %e \n", rr[j], mm[j]);
			fflush(f_oneOverE);
			if (post[i] >= oneOverSqrtE)
			{
				for (j=1; j<=nEpsilon; j++) fprintf(f_oneOverSqrtE, "%f %e \n", rr[j], mm[j]);
				fflush(f_oneOverSqrtE);
			}
		}

	}
	
	fclose(f_oneOverE);
	fclose(f_oneOverSqrtE);

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
	free_dvector(post, 1, nMC);
	free_dvector(P1, 1, nMC);
	free_dvector(P2, 1, nMC);
	free_dvector(P3, 1, nMC);
	free_dvector(P4, 1, nMC);
	free_dvector(P5, 1, nMC);
	free_dvector(Ppts, 1, nparam);

	MPI_Finalize();
}
