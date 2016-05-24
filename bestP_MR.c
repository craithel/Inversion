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

#include "header.h"
#define nMC 10000

double *rhopts; 							//-------------------------------------------------------//  
double *p_SLY, *epsilon_SLY, *rho_SLY;					// Re-initialize GLOBAL variables defined in header file //
double *initM, *initRho, *initEpsilon, *centralP;			//							 //
double r_start, Pedge;							//							 //
int numlinesSLY;							//-------------------------------------------------------//

void main()
{
	int i,j;
	double *post, *P1, *P2, *P3, *P4, *P5;
	double *Ppts, *gamma0_param, *acoef_param;
	double p_ns, *rr, *mm;
	double oneOverE, oneOverSqrtE;
	double max_normP, max=-10000.;
	char buff[256];
	FILE *file;

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
	initRho = dvector(1, nEpsilon);
	initEpsilon = dvector(1, nEpsilon);
	initM = dvector(1, nEpsilon);
	centralP = dvector(1, nEpsilon);

	file = fopen("inversion0_output.txt", "rt");
	fgets(buff, 256, file);
	fgets(buff, 256, file);	

	for (i=1; i<=nMC; i++);
	{
		fscanf(file, "%le %le %le %le %le %le",&post[i], &P1[i], &P2[i], &P3[i], &P4[i], &P5[i]);
		if (post[i] > max)
			max = post[i];
	}
	fclose(file);


	for (i=1; i<=nMC;i++) post[i] /= max;
	
	max_normP = max_array(post,nMC);
	oneOverE = max_normP * exp(-1.);
	oneOverSqrtE = max_normP * exp(-0.5);
	
	edgeConditions(&r_start, &Pedge);
	readinSLY(p_SLY, epsilon_SLY, rho_SLY, &numlinesSLY);
	getRhoPts(rhopts);

	p_ns = bisect_linint(rho_ns, rho_SLY, p_SLY, numlinesSLY);				

	char filename1[260] = "MR_oneOverE.txt";
	FILE *f_oneOverE = fopen(filename1,"w");
	fprintf(f_oneOverE, "File created by bestP_MR.c \n");
	fprintf(f_oneOverE, "All MR are computed for values of (P1, ..., P5) that have posterior likelihoods greater than 1/e \n");
	fprintf(f_oneOverE, "R          M \n");

	char filename2[260] = "MR_oneOverSqrtE.txt";
	FILE *f_oneOverSqrtE = fopen(filename2,"w");
	fprintf(f_oneOverE, "File created by bestP_MR.c \n");
	fprintf(f_oneOverE, "All MR are computed for values of (P1, ..., P5) that have posterior likelihoods greater than 1/sqrt(e) \n");
	fprintf(f_oneOverE, "R          M \n");


	for (i=1; i<=nMC;i++)
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

			for (j=1; j<=nEpsilon; j++) fprintf(f_oneOverE, "%f %f \n", rr[j], mm[j]);
			if (post[i] >= oneOverSqrtE)
				for (j=1; j<=nEpsilon; j++) fprintf(f_oneOverSqrtE, "%f %f \n", rr[j], mm[j]);

		}

	}

	fclose(f_oneOverE);
	fclose(f_oneOverSqrtE);
	
	free_dvector(Ppts, 1, nparam);
	free_dvector(rhopts, 1, nparam);
	free_dvector(gamma0_param, 1, nparam);
	free_dvector(acoef_param, 1, nparam);
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
	free_dvector(post, 1, nMC);
}
