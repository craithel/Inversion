/* C. Raithel, June 2016

  This program finds a suitable starting value for each pressure (P1, ..., P5)
  as well as a good guess for the step size for each pressure. It accomplishes
  both tasks by creating a histogram of pressures at each fiducial density using
  tabular EoS. 

*/

#include "initP.h"
#include "tov.h"
#include "header_useful.h"
#include "../nrutil.h"

#define nBins 6
#define nFiles 42
#define nPgrid 50
#define lines_local 800

static void getBins(double *P, double *bins, double *bins_centered, double *delta_bin);
static void readinEoS(char *filename, char *extdat, double *p, double *eps, double *rho, int *numlines);

extern void initP(double *mean_P1, double *sigma_P1, double *mean_P2, double *sigma_P2, double *mean_P3, double *sigma_P3, double *mean_P4, double *sigma_P4, double *mean_P5, double *sigma_P5)
{
	int i, j, numlines;
	double *P_EoS, *rho_EoS, *eps_EoS;
	double *P1, *P2, *P3, *P4, *P5;
	double *P1_bins, *P2_bins, *P3_bins, *P4_bins, *P5_bins; 
	double *P1_bins_c, *P2_bins_c, *P3_bins_c, *P4_bins_c, *P5_bins_c; 
	double *P1_hist, *P2_hist, *P3_hist, *P4_hist, *P5_hist;
	double *a1, *a2, *a3, *a4, *a5;
	double P1_68, P2_68, P3_68, P4_68, P5_68;
	double Prob1_68, Prob2_68, Prob3_68, Prob4_68, Prob5_68;
	double c1, c2, c3, c4, c5;
	double delta1, delta2, delta3, delta4, delta5;
	double testP;
	char EOSfile[256], EOSext[16];

	P_EoS = dvector(1, lines_local);
	rho_EoS = dvector(1, lines_local);
	eps_EoS = dvector(1, lines_local);

	P1 = dvector(1, nFiles);
	P2 = dvector(1, nFiles);
	P3 = dvector(1, nFiles);
	P4 = dvector(1, nFiles);
	P5 = dvector(1, nFiles);
	
	P1_bins = dvector(1, nBins+1);
	P2_bins = dvector(1, nBins+1);
	P3_bins = dvector(1, nBins+1);
	P4_bins = dvector(1, nBins+1);
	P5_bins = dvector(1, nBins+1);

	P1_bins_c = dvector(1, nBins);
	P2_bins_c = dvector(1, nBins);
	P3_bins_c = dvector(1, nBins);
	P4_bins_c = dvector(1, nBins);
	P5_bins_c = dvector(1, nBins);

	P1_hist = dvector(1, nBins);
	P2_hist = dvector(1, nBins);
	P3_hist = dvector(1, nBins);
	P4_hist = dvector(1, nBins);
	P5_hist = dvector(1, nBins);

	a1 = dvector(1, nPgrid);
	a2 = dvector(1, nPgrid);
	a3 = dvector(1, nPgrid);
	a4 = dvector(1, nPgrid);
	a5 = dvector(1, nPgrid);


	FILE *EOSlist;
	EOSlist = fopen("/gsfs1/xdisk/craithel/EoSfiles/EOSlist.txt","rt");
	
	j=1;
	while (fscanf(EOSlist, "%s %s", &EOSfile, &EOSext) == 2)
	{
		readinEoS(EOSfile, EOSext, P_EoS, eps_EoS, rho_EoS, &numlines);

		P1[j] = bisect_linint(rhopts[2], rho_EoS, P_EoS, numlines); 
		P2[j] = bisect_linint(rhopts[3], rho_EoS, P_EoS, numlines); 	
		P3[j] = bisect_linint(rhopts[4], rho_EoS, P_EoS, numlines); 	
		P4[j] = bisect_linint(rhopts[5], rho_EoS, P_EoS, numlines); 	
		P5[j] = bisect_linint(rhopts[6], rho_EoS, P_EoS, numlines); 	
		j++;
	}

	fclose(EOSlist);

	getBins(P1, P1_bins, P1_bins_c, &delta1);
	getBins(P2, P2_bins, P2_bins_c, &delta2);
	getBins(P3, P3_bins, P3_bins_c, &delta3);
	getBins(P4, P4_bins, P4_bins_c, &delta4);
	getBins(P5, P5_bins, P5_bins_c, &delta5);

	c1 = 1./(delta1*nFiles); 		// normalization factor for each histogram 
	c2 = 1./(delta2*nFiles);
	c3 = 1./(delta3*nFiles);
	c4 = 1./(delta4*nFiles);
	c5 = 1./(delta5*nFiles);

	for (i=1; i<=nFiles; i++) 		//Compute the normalized histogram
	{
		for (j=1; j<=nBins; j++)
		{
			if ( P1[i] >= P1_bins[j] && P1[i] < P1_bins[j+1]) P1_hist[j] += c1; 
			if ( P2[i] >= P2_bins[j] && P2[i] < P2_bins[j+1]) P2_hist[j] += c2; 
			if ( P3[i] >= P3_bins[j] && P3[i] < P3_bins[j+1]) P3_hist[j] += c3; 
			if ( P4[i] >= P4_bins[j] && P4[i] < P4_bins[j+1]) P4_hist[j] += c4; 
			if ( P5[i] >= P5_bins[j] && P5[i] < P5_bins[j+1]) P5_hist[j] += c5; 
		}
	}	


	double step1 = ( max_array(P1_hist, nBins) -1.) / (nPgrid-1.);
	double step2 = ( max_array(P2_hist, nBins) -1.) / (nPgrid-1.);
	double step3 = ( max_array(P3_hist, nBins) -1.) / (nPgrid-1.);
	double step4 = ( max_array(P4_hist, nBins) -1.) / (nPgrid-1.);
	double step5 = ( max_array(P5_hist, nBins) -1.) / (nPgrid-1.);

	double test1 = 1.;
	double test2 = 1.;
	double test3 = 1.;
	double test4 = 1.;
	double test5 = 1.;
	j=1;
	for (j=1; j<= nPgrid; j++)		//Search through a grided (vertically) hist
	{					//to find where the area = 68%
		for (i=1; i<=nBins; i++)
		{
			if (P1_hist[i] >= test1) a1[j] += P1_hist[i];
			if (P2_hist[i] >= test2) a2[j] += P2_hist[i];
			if (P3_hist[i] >= test3) a3[j] += P3_hist[i];
			if (P4_hist[i] >= test4) a4[j] += P4_hist[i];
			if (P5_hist[i] >= test5) a5[j] += P5_hist[i];
		}
		test1 += step1;
		test2 += step2;
		test3 += step3;
		test4 += step4;
		test5 += step5;
	}

	test1 = 1.;
	test2 = 1.;
	test3 = 1.;
	test4 = 1.;
	test5 = 1.;
	for (i=1; i<= nPgrid; i++)						//Search through P-grid areas to find the one closest to 0.68
	{									//Note: a is not monotonic so can't use bisection method
		if ( fabs(0.68 - a1[i]) < test1)
		{
			test1 = fabs(0.68 - a1[i]);
			Prob1_68 = step1*(i-1)+i;			
		}
		if ( fabs(0.68 - a2[i]) < test2)
		{
			test2 = fabs(0.68 - a2[i]);
			Prob2_68 = step2*(i-1)+i;
		}
		if ( fabs(0.68 - a3[i]) < test3)
		{
			test3 = fabs(0.68 - a3[i]);
			Prob3_68 = step3*(i-1)+i;			
		}
		if ( fabs(0.68 - a4[i]) < test4)
		{
			test4 = fabs(0.68 - a4[i]);
			Prob4_68 = step4*(i-1)+i;			
		}
		if ( fabs(0.68 - a5[i]) < test5)
		{
			test5 = fabs(0.68 - a5[i]);
			Prob5_68 = step5*(i-1)+i;			
		}
	}

	test1 = 5000.;
	test2 = 5000.;
	test3 = 5000.;
	test4 = 5000.;
	test5 = 5000.;
	for (i=1; i<=nBins-1; i++)
	{
		if ( fabs(Prob1_68 - P1_hist[i]) < test1)
		{
			test1 = fabs(Prob1_68 - P1_hist[i]);
			P1_68 = P1_bins_c[i];
		}
		if ( fabs(Prob2_68 - P2_hist[i]) < test2)
		{
			test2 = fabs(Prob2_68 - P2_hist[i]);
			P2_68 = P2_bins_c[i];
		}
		if ( fabs(Prob3_68 - P3_hist[i]) < test3)
		{
			test3 = fabs(Prob3_68 - P3_hist[i]);
			P3_68 = P3_bins_c[i];
		}
		if ( fabs(Prob4_68 - P4_hist[i]) < test4)
		{
			test4 = fabs(Prob4_68 - P4_hist[i]);
			P4_68 = P4_bins_c[i];
		}
		if ( fabs(Prob5_68 - P5_hist[i]) < test5)
		{
			test5 = fabs(Prob5_68 - P5_hist[i]);
			P5_68 = P5_bins_c[i];
		}
	} 
	

	int maxP1_index, maxP2_index, maxP3_index, maxP4_index, maxP5_index;
	maxP1_index = max_array_index(P1_hist, nBins);
	maxP2_index = max_array_index(P2_hist, nBins);
	maxP3_index = max_array_index(P3_hist, nBins);
	maxP4_index = max_array_index(P4_hist, nBins);
	maxP5_index = max_array_index(P5_hist, nBins);

	*mean_P1 = P1_bins_c[maxP1_index]; 
	*mean_P2 = P2_bins_c[maxP2_index]; 
	*mean_P3 = P3_bins_c[maxP3_index];
	*mean_P4 = P4_bins_c[maxP4_index];
	*mean_P5 = P5_bins_c[maxP5_index];

	*sigma_P1 = fabs(*mean_P1 - P1_68);
	*sigma_P2 = fabs(*mean_P2 - P2_68);
	*sigma_P3 = fabs(*mean_P3 - P3_68);
	*sigma_P4 = fabs(*mean_P4 - P4_68);
	*sigma_P5 = fabs(*mean_P5 - P5_68);		//For P5, found the RHS 68% level

	FILE *outf;
	outf = fopen("EOShist.txt","w");
	fprintf(outf, "File generated by initP.c \n");
	fprintf(outf, "mean Ppts: %e %e %e %e %e \n", *mean_P1*p_char, *mean_P2*p_char, *mean_P3*p_char, *mean_P4*p_char, *mean_P5*p_char);
	fprintf(outf, "Ppts_68: %e %e %e %e %e \n", P1_68*p_char, P2_68*p_char, P3_68*p_char, P4_68*p_char, P5_68*p_char);
	fprintf(outf, "left-sided sigma: %e %e %e %e %e \n", *sigma_P1*p_char, *sigma_P2*p_char, *sigma_P3*p_char, *sigma_P4*p_char, *sigma_P5*p_char);
	fprintf(outf, "P1 bins     P1 hist      P2 bins     P2 hist       P3 bins      P3 hist        P4 bins       P4 hist      P5 bins        P5 hist\n");
	for (i=1; i<=nBins; i++)
	{
				//print out LEFT hand side of bin
		fprintf(outf, "%e %e %e %e %e %e %e %e %e %e \n", P1_bins[i]*p_char, P1_hist[i]/p_char, P2_bins[i]*p_char, P2_hist[i]/p_char, P3_bins[i]*p_char, P3_hist[i]/p_char, P4_bins[i]*p_char, P4_hist[i]/p_char, P5_bins[i]*p_char, P5_hist[i]/p_char);

				//Print out RIGHT hand side of bin
		fprintf(outf, "%e %e %e %e %e %e %e %e %e %e \n", P1_bins[i+1]*p_char, P1_hist[i]/p_char, P2_bins[i+1]*p_char, P2_hist[i]/p_char, P3_bins[i+1]*p_char, P3_hist[i]/p_char, P4_bins[i+1]*p_char, P4_hist[i]/p_char, P5_bins[i+1]*p_char, P5_hist[i]/p_char);	
	}

	fclose(outf);

	free_dvector(P1, 1, nFiles);
	free_dvector(P2, 1, nFiles);
	free_dvector(P3, 1, nFiles);
	free_dvector(P4, 1, nFiles);
	free_dvector(P5, 1, nFiles);
	free_dvector(P_EoS, 1, lines_local);
	free_dvector(rho_EoS, 1, lines_local);
	free_dvector(eps_EoS, 1, lines_local);
	free_dvector(P1_bins, 1, nBins+1);
	free_dvector(P2_bins, 1, nBins+1);
	free_dvector(P3_bins, 1, nBins+1);
	free_dvector(P4_bins, 1, nBins+1);
	free_dvector(P5_bins, 1, nBins+1);
	free_dvector(P1_bins_c, 1, nBins);
	free_dvector(P2_bins_c, 1, nBins);
	free_dvector(P3_bins_c, 1, nBins);
	free_dvector(P4_bins_c, 1, nBins);
	free_dvector(P5_bins_c, 1, nBins);
	free_dvector(P1_hist, 1, nBins);
	free_dvector(P2_hist, 1, nBins);
	free_dvector(P3_hist, 1, nBins);
	free_dvector(P4_hist, 1, nBins);
	free_dvector(P5_hist, 1, nBins);
	free_dvector(a1, 1, nPgrid);
	free_dvector(a2, 1, nPgrid);
	free_dvector(a3, 1, nPgrid);
	free_dvector(a4, 1, nPgrid);
	free_dvector(a5, 1, nPgrid);
}


static void readinEoS(char *filename, char *extdat, double *p, double *eps, double *rho, int *numlines)
/* Read in tabulated data for the low-density regime, using any tabulated EoS. */
{

	int i=1,j;
	FILE *file;
	file = fopen(filename,"rt");

	if (strcmp(extdat,"Y") == 0)							//".dat" file types
	{
		while ( fscanf(file, "%le %le %le", &rho[i], &p[i], &eps[i]) == 3) i++;
		*numlines = i-1;
		for (j=1; j<=*numlines; j++)
		{
			p[j] = p[j]*clight*clight/p_char;
			rho[j] /= rho_char;
			eps[j] /= rho_char;
		}
	}
	else										//"EOS." file types
	{	
		while ( fscanf(file,"%le %le %le", &rho[i], &eps[i], &p[i]) == 3 ) i++;	
		*numlines = i-1;							
		for (j=1; j<=*numlines; j++)
		{
			p[j] = pow(10.0,p[j]) / p_char;	
			eps[j] = pow(10.0, eps[j]) / rho_char;				//convert to dim'less units
			rho[j] = pow(10.0, rho[j])*(1.659e-24) /  rho_char;
		}
	}
	
	fclose(file);								//Close EoS file

}

static void getBins(double *P, double *bins, double *bins_centered, double *delta_bin)
{
	int i;
	double lbound, ubound, half_delta;
	lbound = min_array(P, nFiles);
	ubound = max_array(P, nFiles);

	*delta_bin = (ubound - lbound)/(1.*nBins);

	bins[1] = lbound;
	for (i=2; i<=nBins+1; i++) bins[i] = bins[i-1] + *delta_bin;
	
	half_delta = *delta_bin*0.5;	
	for (i=1; i<=nBins; i++) bins_centered[i] = bins[i] + half_delta;

}
