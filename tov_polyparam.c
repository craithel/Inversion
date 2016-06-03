/* C. Raithel, May 2016
 This MODULE is a version of tov_polyparam_segEoS_realistic_noPT.c 
 that has been generalized to compute the M, R, and moment of inertia
 using our optimized parametric 5-polytrope EoS.
 The module takes in as parameters the pressures (Ppts) at and values of
 the five fiducial densities. It also uses the global variables of the 
 SLY EoS for the low-density regime.

*/

#include "tov.h"

//static double findnu(int j);							//Prototype func to calculate nu, given an initial guess ~1
//static double scaleI(int j);							//Prototype func to scale I_trial
static double EOSdensity(double pressure, int useparam);	   		//Prototype func for the EOS, P=P(rho)	
static double derivs(int i, double r, double array[]);			 	//Prototype func for 6 ODEs: dP/dr, dM/dr, dv/dr and (3 for) dI/dr
static void rkdriver(double r1,double p_edge); 	//Prototype func for Runge-Kutta integration driver function

//static double *centralP;							//Initialize global array to save central pressures
//static double (*y_v);						   		//Initialize global array to save solutions for nu (from metric)
//static double (*y_f);						   		//Initialize global array to save solutions for f = 1-w/Omega
//static double (*y_i);						   		//Initialize global array to save solutions for f = 1-w/Omega
static int iter;
static double *Ppts;							
static double *gamma0_param, *acoef_param;					
static double *y_m, *rr;		


extern void tov(double Ppts_in[], double *rr_in, double *y_m_in, double *gamma0_param_in, double *acoef_param_in)
/* Main driver function to get M and  R (moment of inertia disabled)
   by solving the TOV equations (+ ODEs for I)
*/
{
	int i,j,k, EoSindex, index;	

	Ppts = dvector(1,nparam);				
	gamma0_param = gamma0_param_in;			//Set the global pointer for gamma0_param to the location of gamma0 from the main() func in inversion	
	acoef_param = acoef_param_in;
	rr = rr_in;
	y_m = y_m_in;
	
	for (i=1; i<= nparam; i++)
		Ppts[i] = Ppts_in[i];													
	
	getGammaA(Ppts, gamma0_param,acoef_param);	//Find Gamma and a_coef along each segment, given Ppts

	rkdriver(r_start, Pedge);			//Call Runge-Kutta integrating function


	//for (j=1;j<=nEpsilon;j++)  inertia[j] = scaleI(j);			//Scale the trial moment of inertia, as in Kalogera & Psaltis 99

	free_dvector(Ppts, 1, nparam);
	//free_dvector(gamma0_param, 1, nparam);
	//free_dvector(acoef_param, 1, nparam);
	//free_dvector(initEpsilon, 1, nEpsilon);
	//free_dvector(initM, 1, nEpsilon);
	//free_dvector(y_i,1,nEpsilon);
	//free_dvector(y_v,1,nEpsilon);
	//free_dvector(y_f,1,nEpsilon);
	//free_dvector(centralP,1,nEpsilon);

}

void rkdriver(double r1, double p_edge)
/* Fourth-order Range-Kutta driver.
 Starting from initial values vstart[1..nrho] known at r1, use fourth-order RK to advance
 by increments calculated depending on the local environment. Continue advancing until the
 edge pressure is negligible wrt the central pressure. The routine "derivs" evaluates the 
 derivatives dM/dR, dP/dr, and dI/dr by which to step the mass, pressure, and mom of I, 
 respectively. The density at the new increment is calculated using the EoS for density. 
 Results are stored in the global variables: y_m[1..nrho][1..nstep+1], y_p[1..nrho][1..nstep+1], 
 y_rho[1..nrho][1..nstep+1], y_I[1..nrho][1..nstep+1], and rr[1..nstep+1].
*/
{
	void rk4(double y_eps, double y[],				//Prototype for 4th-order RK stepping function
		 double dydr[], double r, double h, double yout[],
		 double *yout_eps);
		
	int i,j,k,index;
	double r,hthisstep,h_Pstep,h_Mstep;
	double v_eps, vout_eps;
	double *v, *vout;
	double *dv;							//Order ALWAYS: M, P, v, f, cassi, I

	v=dvector(1,nODEs);
	vout=dvector(1,nODEs);
	dv=dvector(1,nODEs);

	//y_v=dvector(1,nEpsilon);					//Assign memory for nu
	//y_f=dvector(1,nEpsilon);					//Assign memory for f
	//y_i=dvector(1,nEpsilon);					//Assign memory for I (unscaled)
	//centralP=dvector(1,nEpsilon);

	for (i=1; i<=nEpsilon; i++)					//Step through all values of rho_c
	{
	  //for (index=1; index<=2; index++)				//Loop through each rho_c twice, to determine starting value of nu
	  //{	
		r=r1;							//First step located at initial radius
		v_eps=initEpsilon[i];           			//Load initial values for density, rho

		v[1]=initM[i];             				//Load initial values for M
		v[2]=centralP[i];
		/*if (index==1) 
			v[3]=1.0;					//Load initial value for nu at center = 0
		else
			v[3]=findnu(i);

		v[4]=1.0;						//Initial f at center -- of order unity, will be scaled!
		v[5]=0.0;						//Initial cassi = df/dr = 0 (at center)
		v[6]=0.0;						//Load initial mom of I
		*/
		k=1;	
		while (k <= nstep && v[2] >= p_edge*centralP[i])		//Keep stepping until P_edge (v_p) is negligible or exceed nstep	
		{
			h_Pstep= (v[2]*r*r) /						//Define stepsize from minimum of dP/dr 
				 ( (v_eps+v[2])*(v[1] + 3.0*r*r*r*v[2])*MoverR );  	// and dM/dr, calculated the the local environment    
			h_Mstep = v[1] / (3.0*r*r*v_eps);				// at each step
			hthisstep = min(fabs(h_Pstep), fabs(h_Mstep))*hscale;

			for (j=1;j<=nODEs;j++) dv[j]=derivs(j,r,v);      //,v[3],v[4],v[5]); 	//Get initial derivs for all 6 funcs
			rk4(v_eps, v, dv, r, hthisstep, vout, &vout_eps);
			r=r+hthisstep;	    					//Increase the radius increment		

		   	v_eps=vout_eps;	    					//Save updated terms
			v[1]=vout[1];	    					//mass				
			v[2]=vout[2];	    					//pressure	
			/*v[3]=vout[3];	    					//nu			
			v[4]=vout[4];	    					//f_trial	
			v[5]=vout[5];	    					//cassi		
			v[6]=vout[6];*/						//I_trial
			k+=1;							//Increase the stepper
			if (k-1==nstep && v[2] >= p_edge*centralP[i])		//Verify that we've actually reached edge w/in nsteps
				printf("Failed to reach edge for rho_c: %d\n",i);	//Print warning if fail to reach edge
		}
		rr[i] = r;							//Save the total neutron star radius,
		y_m[i] = v[1];							//mass
		//y_v[i] = v[3];							//nu (so it can be scaled)
		//y_f[i] = v[4];							//f_trial
		//y_i[i] = v[6];							//and I_trial
		//if (index==1) iter=k;
	  //}
	}

	free_dvector(v,1,nODEs);
	free_dvector(vout,1,nODEs);
	free_dvector(dv,1,nODEs);

}

void rk4(double y_eps, double y[], double dydr[], double r, double h, 
	double yout[], double *yout_eps)
/* Fourth-order Range-Kutta method, modified from Numerical Recipes.
 Given values for the variables y[1..n] and their derivs dydr[1..n] known at r,
 use the 4th-order RK method to advance the solution over an interval h and return
 the incremented values yout[1..n], which need not be a distinct array from y.
 The user supplies the routine derivs(x,y,dyxy) which returns the derivatives dydx
 at x.
*/
{
	int j;
   	double rh, hh, h6;
	double yt_eps;	
	double *dym, *dyt;						//Order ALWAYS: M, P, v, f, cassi, I
	double *yt;
	dym=dvector(1,nODEs);
	dyt=dvector(1,nODEs);
	yt=dvector(1,nODEs);

	hh = h*0.5;              							//Half-step increment
    	h6 = h/6.0;             							//1/6th step increment
    	rh = r+hh;           								//Midpoint x-value
	
	for (j=1;j<=nODEs;j++) yt[j] = y[j] + hh*dydr[j];				//First step
	for (j=1;j<=nODEs;j++) dyt[j]=derivs(j,rh,yt);				 	//Second step
	for (j=1;j<=nODEs;j++) yt[j] = y[j] + hh*dyt[j];
	for (j=1;j<=nODEs;j++) dym[j]=derivs(j,rh,yt);					 //Third step			 						
	for (j=1;j<=nODEs;j++) 
	{
		yt[j] = y[j] + h*dym[j];
		dym[j] += dyt[j];
	}	
	for (j=1;j<=nODEs;j++) dyt[j]=derivs(j,r+h,yt);	//Fourth step
	for (j=1;j<=nODEs;j++) yout[j] = y[j]+h6*(dydr[j]+dyt[j]+2.0*dym[j]);		//Accumulate increments with proper weights
	if (yout[2] <= Ppts[1]) 							//Calculate final energy density at 3rd step from inverse EOS
		*yout_eps = EOSdensity(yout[2],0);					//Using SLY if in the low-density regime
	else
		*yout_eps = EOSdensity(yout[2],1);					//Otherwise use parametrized EOS		
	free_dvector(dym,1,nODEs);
	free_dvector(dyt,1,nODEs);
	free_dvector(yt,1,nODEs);
}

extern double EOSpressure(double massdensity, double Ppts[], double gamma0_param[], int useparam)
/* Find the EoS pressure given a density, by searching through the tabulated
 values of pressure and density. Use bisection method to search.	
*/
{
	double pressure;
	int xmid;							//Index at midpoint
	int j, a=1,b=numlinesSLY;					//Index bounds for whole interval
	double m,intercept;						//Slope and int for linear interpolation

	if (useparam==0)						//If flag "useparam" set to 0 , find pressure using ~~~SLy~~~
	{
		do							//Do a bisection search to find the pressure from the energy density
		{
			xmid=(a+b)/2;						//Calculate index at midpt
			if (massdensity < rho_SLY[xmid])			//If the density is lower than tabulated value at midpt
				b = xmid;					//set new upper bound to be the midpoint
			else 							//Otherwise
				a = xmid;					//set new lower bound to be the midpoint	
		}
		while (fabs(b-a) > 1);						//until converged to within 2 tabulated values (index diff=1)

		m= (p_SLY[b] - p_SLY[a]) / (rho_SLY[b] - rho_SLY[a]);		 //Calculate slope to linearly interpolate btwn a and b
		intercept = p_SLY[a] - m*rho_SLY[a];				//Calculate y-intercept
		pressure = m*massdensity + intercept;				//get pressure

		if ( pressure > p_SLY[numlinesSLY] || pressure < p_SLY[1] )	//check results of bisection search
		{
			printf("Warning: bisection method found P %e at edge of range %d %d \n",pressure*p_char,a,b);
			fflush(stdout);
		}
	}
	else									//Otherwise use the parametrized EoS
	{
		for (j=1;j<=nparam-1;j++)
		{
			if (j==1 && massdensity > rhopts[j] && massdensity <= rhopts[j+1])
				pressure = Ppts[j+1]*pow(massdensity/rhopts[j+1], gamma0_param[j]);	
			if (j>1 && massdensity > rhopts[j] && massdensity <= rhopts[j+1] )
				pressure = Ppts[j]*pow(massdensity/rhopts[j], gamma0_param[j]);	
			if (j>1 && massdensity > rhopts[nparam-1])
				pressure = Ppts[nparam-1]*pow(massdensity/rhopts[nparam-1], gamma0_param[nparam-1]);
		}
	}

	return pressure;
} 

double EOSdensity(double pressure, int useparam)
/* Find the EoS density given a pressure, by searching through the tabulated
   values of pressure and density. Use bisection method to search.	
*/
{
	double massdensity, energydensity;
	int xmid,index;							//Index at midpoint
	int j, a=1,b=numlinesSLY;						//Interval bounds
	double m,intercept;						//Slope and int for linear interpolation

	if (useparam==0)						//If flag "useparam" set to 0, find density from full EoS
	{	
		do 
		{
			xmid=(a+b)/2;						//Calculate index at midpt
			if (pressure < p_SLY[xmid])				//If the pressure is lower than tabulated value at midpt
				b = xmid;					//set new upper bound to be the midpoint
			else 							//Otherwise
				a = xmid;					//set new lower bound to be the midpoint	
		} while (fabs(b-a) > 1);					//until converged to within 2 tabulated values (index diff=1)

		if (p_SLY[b] != p_SLY[a])					//If we're not in a flat portion of the segmented EoS,
		{
			m= (rho_SLY[b] - rho_SLY[a]) / (p_SLY[b] - p_SLY[a]);	//Calculate slope to linearly interpolate btwn a and b
			intercept = rho_SLY[a] - m*p_SLY[a];			//Calculate y-intercept
			massdensity = m*pressure + intercept;
		} else
			massdensity = (rho_SLY[b] + rho_SLY[a]) /2.0;		//Otherwise, just take average of nearest 2 density points

		if (massdensity > rho_SLY[numlinesSLY] || massdensity - rho_SLY[1] < -5.0e-6 )
		{	printf("Warning: bisection method found Rho at edge of range %d %d \n", a, b);
			printf("pressure: %e\n", pressure*p_char);
			printf("massdensity: %e \n",massdensity*rho_char);
			fflush(stdout);
			//exit(0);
		}

		energydensity = findEps0_inSLY(massdensity);			//search in SLy for epsilon

	}
	else								//Otherwise use parametrized EoS
	{
		for (j=1; j<=nparam-1; j++)
		{
			if (j==1 && pressure > Ppts[j] && pressure <= Ppts[j+1])
				massdensity = rhopts[j+1]*pow(pressure/Ppts[j+1], 1.0/gamma0_param[j]);
			if (j>1 && (pressure > Ppts[j] && pressure <= Ppts[j+1]) )
				massdensity = rhopts[j]*pow(pressure/Ppts[j], 1.0/gamma0_param[j]);
			if (j>1 && pressure > Ppts[nparam-1])
				massdensity = rhopts[nparam-1]*pow(pressure/Ppts[nparam-1], 1.0/gamma0_param[nparam-1]);
		}

		if (massdensity <= rhopts[1])
			energydensity = findEps0_inSLY(massdensity);		//if lower than rhopts[1], search in SLy for epsilon
		else
			energydensity = param_massToEnergy(massdensity,Ppts,gamma0_param,acoef_param);	//Convert mass to energy density
	}


	return energydensity;
}

  
double derivs(int j, double r, double array[])  
/* Calculate dimensionless derivatives for dM/dr, dP/dr, and dI/dr
   Order ALWAYS: M, P, v, f, cassi, I
*/
{

	double mass, pressure, nu, f, cassi;
	double density;
	
	mass = array[1];
	pressure = array[2];
	if (nODEs > 2)
	{
		nu = array[3];
		f = array[4];
		cassi = array[5];
	}

	if (pressure <= Ppts[1])				 	//Find the energy density for the given pressure
 		density=EOSdensity(pressure,0); 			//Using SLy if low-density
	else
		density=EOSdensity(pressure,1); 			//And PARAMETRIZED EoS otherwise	

	if (j==1)							//Return dimensionless form of dM/dr
	{	
		return 3.0*r*r*density;
	}	
	if (j==2)
	{	
		return -MoverR*(density+pressure)*			//Return dimensionless form of dP/dr					
	         (mass + 3.0*r*r*r*pressure) / 
	         (r*r - 2.0*r*mass*MoverR) ;	
	}	
	if (j==3)
	{	
		return 2.0*MoverR*(mass + 3.0*r*r*r*pressure) /		//Return dimensionless form of dv/dr	
		 (r*r - 2.0*r*mass*MoverR) ;
	}	
	if (j==4)
	{	
		return cassi*exp(nu/2.0) / ( r*r*r*r* 			//Return dimensionless form of df/dr	
		 pow(1.0-2.0*MoverR*mass/r,0.5) );
	}
	if (j==5)
	{	

		return 12.0*exp(-nu/2.0)*MoverR*			//Return dimensionless form of dcassi/dr
		 f * r*r*r*r * (pressure + density)  /
		 pow(1.0 - 2.0*MoverR*mass/r,0.5) ;
	}
	if (j==6)
	{
		return 2.0*(density+pressure)*f*			//Return dimensionless form of dI/dr
		 r*r*r*r * exp(-nu/2.0) /
		 pow(1.0 - 2.0*MoverR*mass/r,0.5) ;
	}
	return 0.;
}

//double scaleI(int j)
/* Calculate the scaled moment of inertia, using the trial I and f values
   (as in Eq. 16 in KP99) AND convert back into CGS units
*/
/*{
	return y_i[j]*(Msolar*1.0e10) / ( y_f[j] + 
		  2.0*y_i[j]*(Msolar*1.0e10)*Ggrav/(clight*clight*rr[j]*rr[j]*rr[j]*1.0e15) );
}*/


//double findnu(int j)
/* Calculate the scaled value for nu at the center (r=0), to use as nu_initial
   in the second iteration of the main program. Use boundary condition:
   nu(R_ns) = ln[1 - 2 M_ns/R_ns]
*/
/*{
	double nu_R;	
 	nu_R=log(1.0-2.0*y_m[j]*Msolar*Ggrav/(rr[j]*1.0e5*clight*clight) );	//nu at surface of star
	return 1.0 - y_v[j] + nu_R;						//New nu_c = Old nu_c - (nu_surface_actual - nu_surface_required)

}*/

extern double param_massToEnergy(double massdensity, double Ppts[], double gamma0_param[], double acoef_param[])
/* Given a mass density, return an energy density, using only the parametrization points
   and the corresponding a's, gamma's, etc. for the parametrization (i.e. do not invoke
   the full EoS).
*/
{
	int j, index;
	double energy;

	if (massdensity > rhopts[nparam-1])		//If rho > 2nd to last param. point, then you're in the final parametrized segment
		index=nparam-1;
	else						//Otherwise, determine which segment of the EOS you're in
	{
		for (j=1; j <= nparam-1; j++) 
			if (massdensity > rhopts[j] && massdensity <= rhopts[j+1]) index=j;									
	}

	if (fabs(gamma0_param[index]-1.0) < 0.01)				//If we're in a shallow segment (gamma=1), use special case of eps-rho conversion
	{
		energy = (1.0 + acoef_param[index])*massdensity + (Ppts[index+1]/rhopts[index+1]) *	
			  massdensity*log(massdensity) ;

	} else									//Otherwise, use usual eps-rho conversion (as in OP09) to
	{									//convert mass density to energy density for that region
		energy = (1.0 + acoef_param[index])*massdensity + (Ppts[index+1]/(gamma0_param[index]-1.0))*
			 pow(massdensity/rhopts[index+1], gamma0_param[index]) ;
	}

	return energy;
}


extern double findEps0_inSLY(double rho0)
/* Use bisection method to search for the energy_density that corresponds to our rho_0
   in the tabulated SLY data
*/
{

	int xmid;							//Index at midpoint
	int a=1,b=numlinesSLY;						//Index bounds for whole interval
	double m,intercept, eps0;					//Slope and int for linear interpolation

	do
	{
		xmid=(a+b)/2;
		if (rho0 < rho_SLY[xmid])
			b = xmid;
		else
			a = xmid;
	} 
	while (fabs(b-a) > 1);	

	m= (epsilon_SLY[b] - epsilon_SLY[a]) / (rho_SLY[b] - rho_SLY[a]);	//Calculate slope to linearly interpolate btwn a and b
	intercept = epsilon_SLY[a] - m*rho_SLY[a];				//Calculate y-intercept
	eps0 = m*rho0 + intercept;
	return eps0;
}


extern void getGammaA(double Ppts[], double *gamma0_param, double *acoef_param)
/*
  Calculate the gamma and coefficient a (the integration constant in eps(rho) )
  along each segment of the ~polytropic~ EoS
*/
{
	int j;
	double epsilon0_param;

	epsilon0_param = findEps0_inSLY(rhopts[1]);

	for (j=1;j<=nparam-1;j++)							//Find a and gamma for each segment of the parametrized EoS
	{
		gamma0_param[j] = log10(Ppts[j+1]/Ppts[j]) / log10(rhopts[j+1]/rhopts[j]) ;


		if (j==1)
		{
			if (fabs(gamma0_param[j]-1.0) < 0.01)					//if gamma=1, use special case of eps-rho relation
			{
				acoef_param[1] = (epsilon0_param/rhopts[1]) - 1.0 - 
			  	 		Ppts[2]*log(rhopts[1])/rhopts[2];
			} else 									//otherwise, define 'a' as in OP09
			{
				acoef_param[1] = (epsilon0_param/rhopts[1]) - 1.0 - 
			  	 		Ppts[2]*pow(rhopts[1]/rhopts[2],gamma0_param[1]) /
			   	 		( (gamma0_param[1]-1.0)*rhopts[1] );
			}
		}
		else
		{
			if ( fabs(gamma0_param[j]-1.0) < 0.01)					//if gamma=1, use special case of eps-rho relation
			{
				acoef_param[j] =  ( param_massToEnergy(rhopts[j],Ppts,gamma0_param,acoef_param) / rhopts[j] ) - 1.0 - 
						Ppts[j]*log(rhopts[j]) / rhopts[j] ; 
			} else									//otherwise, define 'a' as in OP09
			{
				acoef_param[j] =  ( param_massToEnergy(rhopts[j],Ppts,gamma0_param,acoef_param) / rhopts[j] ) - 1.0 - 
						Ppts[j] / (rhopts[j]*(gamma0_param[j]-1.0) );
			}
		}	
	}
}
