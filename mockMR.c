/* C. Raithel, May 2016

 This program gathers masses and radii from the EoS SLy
 then dithers their values and assigns some uncertainties

*/

//#include "header.h"
#include "mockMR.h"
#include "mtwist-1.5/mtwist.h"

static double *mass, *radius;
static int guess_lines, numlines;
static void readinMR(double mass[], double radius[], double I[]);
void Igaussian(double *I_step, double I_sigma);
void MRgaussian(double *m_step, double m_sigma, double *r_step, double r_sigma, double mean);

extern void getMR(int realization)
{
	int i, j, counter=1;
	unsigned int seed;
	double Mdither, Rdither;
	double *mWant, rWant, mDelta;
	double *Ifromfile;
	guess_lines = 100;
	mass = dvector(1, guess_lines);
	radius = dvector(1, guess_lines);
	Ifromfile = dvector(1, guess_lines);
	mWant = dvector(1,nData+1);

	readinMR(mass, radius,Ifromfile);
	mDelta = (max_array(mass, guess_lines) - 1.2)/nData; 
	mWant[1] = 1.2;
	for (i=2; i<=nData; i++) mWant[i] = mWant[i-1]+mDelta;

	//seed = mt_goodseed();
	seed =942700166;
	mt_seed32new(seed);
	

	char name[260];
	sprintf(name, "MRvals_SLY_%d_I.txt",realization);
	FILE *f = fopen(name, "w");
	fprintf(f, "MR values taken from SLy and dithered by a random value, drawn from the gaussian distribution of M, R uncertainties \n");
	fprintf(f, "seed: %u \n", seed);

	/*
	FILE *fUnd = fopen("MRvals_22213_15raw.txt", "w");
	fprintf(fUnd, "Undithered data for the 15 pts on ALF2\n");
	fprintf(fUnd, "R      M \n");
	*/

	moi_EOS = bisect_linint(1.338, mass, Ifromfile, 1, guess_lines);	
	Igaussian(&sigma_moi, 0.1*moi_EOS);
	moi_EOS += sigma_moi;			//dither by 10%
	fprintf(f, "I:  %e   %e \n", moi_EOS, sigma_moi);

	fprintf(f, "M     M_sigma     R    R_sigma  \n");
	fflush(f);

	for (i=1; i<=numlines; i++)
	{	
		//if ( (i%4) == 0 && mass[i] > 0.5 && counter <= nData)	//SLy condition including low masses (down to ~0.5)
		//if ( (i%3)==0 && mass[i] > 1.2 && counter <=nData) // NEW (as of 9/1) mpa1 condition
		//if ( counter <=nData) // soft (#255) EOS condition
		//if ( (i%3) == 0 && mass[i] > 1.2 && counter <= nData)	//SLy condition - 10 pts
		//if ( (mass[i] > 1.2 && mass[i] < 1.7 ) || (mass[i] >= 1.7 && (i%2)==0)  && counter <= nData)	//EOS A condition - 15 pts
		//if ( (mass[i] > 1.2 && mass[i] < 1.4) || (mass[i] >=1.4 && (i%2)==0) && counter <= nData)	//SLy condition - 15 pts OLD
		//if ( (mass[i] > 1.2 && mass[i] < 1.6) || (mass[i] >=1.6 && (i%2)==0) && counter <= nData)	//EOS 29491 condition - 15 pts
		//{
	//for (i=1; i<=nData; i++)
	//{
		if ( (mass[i] > 1.2 && i%2==0) && counter <= nData)	//SLy condition - 15 pts
		{
			m_sigma[counter] = 0.1; // 0.05; //mass[i]/15.;
			r_sigma[counter] = 0.5; //0.25; //radius[i]/20.;

			MRgaussian(&Mdither, m_sigma[counter], &Rdither, r_sigma[counter], 0.); 

			//rWant =bisect_linint(mWant[i], mass, radius, 1, guess_lines);  		//find desired M in R
		
			m_data[counter] = mass[i] + Mdither; //mWant[i] + Mdither;  //mass[i] + Mdither;
			r_data[counter] = radius[i] + Rdither; //rWant + Rdither; 

			//fprintf(fUnd, "%f %f \n", rWant, mWant[i]);
			fprintf(f, "%f %f %f %f \n", m_data[counter], m_sigma[counter], r_data[counter], r_sigma[counter]);
			counter+=1;
		}
	}
	fclose(f);
	//fclose(fUnd);
	free_dvector(mass,1,guess_lines);
	free_dvector(radius,1,guess_lines);
	free_dvector(mWant,1,guess_lines);
	free_dvector(Ifromfile, 1, guess_lines);
}
void Igaussian(double *I_step, double I_sigma)
/* Draw a random number from a Gaussian distribution */
{
	double buf;
	double rand1 = mt_drand();
	double rand2 = mt_drand();

	*I_step = sqrt(-2.*log(rand1))*cos(2.*M_PI*rand2)*I_sigma + 0.;

}

void MRgaussian(double *m_step, double m_sigma, double *r_step, double r_sigma, double mean)
/* Draw a random number from a Gaussian distribution */
{
	double buf;
	double rand1 = mt_drand();
	double rand2 = mt_drand();
	double rand3 = mt_drand();
	double rand4 = mt_drand();

	*m_step = sqrt(-2.*log(rand1))*cos(2.*M_PI*rand2)*m_sigma + mean;
	buf = sqrt(-2.*log(rand1))*sin(2.*M_PI*rand2)*m_sigma + mean;

	*r_step = sqrt(-2.*log(rand3))*cos(2.*M_PI*rand4)*r_sigma + mean;
	buf = sqrt(-2.*log(rand3))*sin(2.*M_PI*rand4)*r_sigma + mean;

}

static void readinMR(double mass[], double radius[], double I[])
/* Read in tabulated data for the low-density regime, using the EoS Sly. */
{

	int i=1,j;
	char buff[512];
	char ep[guess_lines], rhoc[guess_lines], pc[guess_lines];

	FILE *file;
 	//file = fopen("/extra/craithel/optimal_param/noPT/seg/testTOV_15_g15_sly/tov_22213_nopt.txt","rt");
 	//file = fopen("/extra/craithel/EoSfiles/MRfiles/tov_alf2.txt","rt");
	file = fopen("/extra/craithel/tov_sly.txt","rt");
	fgets(buff, 512, file);
	fgets(buff, 512, file);
	fgets(buff, 512, file);
	

	while ( fscanf(file,"%s %s %s %le %le %le", &ep[i], &rhoc[i], &pc[i], &radius[i], &mass[i], &I[i]) == 6 ) i++;	
	numlines=i;
	fclose(file);							

}
