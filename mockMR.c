/* C. Raithel, May 2016

 This program gathers masses and radii from the EoS SLy
 then dithers their values and assigns some uncertainties

*/

//#include "header.h"
#include "mockMR.h"
#include "mtwist-1.5/mtwist.h"

static double *mass, *radius;
static int guess_lines, numlines;
static void readinMR(double mass[], double radius[]);

extern void getMR()
{
	int i, counter=1;
	guess_lines = 100;
	mass = dvector(1, guess_lines);
	radius = dvector(1, guess_lines);

	readinMR(mass, radius);

	FILE *f = fopen("MRvals.txt", "w");
	fprintf(f, "MR values taken from SLy and dithered by a random value \n");
	fprintf(f, "M     M_sigma     R    R_sigma  \n");

	for (i=1; i<=numlines; i++)
	{	
		if ( (i%3) == 0 && mass[i] > 1.2 && counter <= nData)	
		{
			if ((i%2) ==0)
			{								//In half the cases,
				m_data[counter] = mass[i]+mt_drand()/10.;		//add noise: MT random double in [0,1) -> [0,0.1)  
				r_data[counter] = radius[i]+mt_drand()/10.;		//add noise: MT random double in [0,1) -> [0,0.1) ;
			} else
			{								//In other half,
				m_data[counter] = mass[i]-mt_drand()/10.;		//add NEGATIVE offset: MT random double in [0,1) -> [0,0.1)  
				r_data[counter] = radius[i]-mt_drand()/10.;		//add NEGATIVE offset: MT random double in [0,1) -> [0,0.1) ;
			}
			m_sigma[counter] = mass[i]/15.;
			r_sigma[counter] = radius[i]/20.;

			fprintf(f, "%f %f %f %f \n", m_data[counter], m_sigma[counter], r_data[counter], r_sigma[counter]);
			counter+=1;

		}

	}
	fclose(f);
	free_dvector(mass,1,guess_lines);
	free_dvector(radius,1,guess_lines);
}

static void readinMR(double mass[], double radius[])
/* Read in tabulated data for the low-density regime, using the EoS Sly. */
{

	int i=1,j;
	char buff[512];
	char ep[guess_lines], rhoc[guess_lines], pc[guess_lines], I[guess_lines];

	FILE *file;
	file = fopen("/gsfs1/xdisk/craithel/tov_sly.txt","rt");
	fgets(buff, 512, file);
	fgets(buff, 512, file);
	fgets(buff, 512, file);

	while ( fscanf(file,"%s %s %s %le %le %s", &ep[i], &rhoc[i], &pc[i], &radius[i], &mass[i], &I[i]) == 6 ) i++;	
	numlines=i;
	fclose(file);							

}
