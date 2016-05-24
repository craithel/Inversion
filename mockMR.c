/* C. Raithel, May 2016

 This program gathers masses and radii from the EoS SLy
 then dithers their values and assigns some uncertainties

*/

#include "header.h"
#include "mtwist-1.5/mtwist.h"

static double *mass, *radius;
static int numlines;
static void readinMR(double mass[], double radius[]);

extern void getMR()
{
	int i, counter=1;
	mass = dvector(1, lines);
	radius = dvector(1, lines);

	readinMR(mass, radius);

	for (i=1; i<=numlines; i++)
	{	
		if ( (i%3) == 0 && mass[i] > 1.2 && counter <= nData)	
		{
			m_data[counter] = mass[i]+mt_drand()/10.;		//add noise: MT random double in [0,1) -> [0,0.1)  
			m_sigma[counter] = mass[i]/15.;
			r_data[counter] = radius[i]+mt_drand()/10.;		//add noise: MT random double in [0,1) -> [0,0.1) ;
			r_sigma[counter] = radius[i]/20.;
			counter+=1;
		}

	}
	
	free_dvector(mass,1,lines);
	free_dvector(radius,1,lines);
}

static void readinMR(double mass[], double radius[])
/* Read in tabulated data for the low-density regime, using the EoS Sly. */
{

	int i=1,j;
	char buff[512];
	char ep[lines], rhoc[lines], pc[lines], I[lines];

	FILE *file;
	file = fopen("/gsfs1/xdisk/craithel/tov_sly.txt","rt");
	fgets(buff, 512, file);
	fgets(buff, 512, file);
	fgets(buff, 512, file);

	while ( fscanf(file,"%s %s %s %le %le %s", &ep[i], &rhoc[i], &pc[i], &radius[i], &mass[i], &I[i]) == 6 ) i++;	
	numlines=i;
	fclose(file);							

}
