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
#include "tov_dev.h"
#include "bestPMR.h"

#define nFiles 369
#define nMCperfile 25600
#define nMC 9472000  //nFiles*nMCperfile

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      FILE *ferr = fopen("gpu.err","w");
      fprintf(ferr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      fclose(ferr);
      if (abort) exit(code);
   }
}

#if __cplusplus
extern "C" {
#endif
__global__ void getMR_fromPs(int nn, double *rr_super, double *mm_super, double *P1, double *P2, double *P3, double *P4, double *P5);
__device__ void tov_dev(double *rr_in, double *y_m_in,double *Ppts);
__device__ double findEps0_inSLY(double rho0);
__device__ void getGammaA(double Ppts[], double *gamma0_param, double *acoef_param);
__device__ void initCond_dev(double *initM, double *initRho, double *initEpsilon, double *centralP, double *Ppts, double *gamma0_param, double *acoef_param);
__device__ double param_massToEnergy(double massdensity, double Ppts[], double gamma0_param[], double acoef_param[]);
__device__ static double EOSdensity(double pressure, double *Ppts, double *gamma0_param, double *acoef_param, int useparam);	   		//Prototype func for the EOS, P=P(rho)	
__device__ double EOSpressure(double massdensity, double Ppts[], double gamma0_param[], int useparam);
__device__ static double derivs(double *Ppts, double *gamma0_param, double *acoef_param, int i, double r, double array[]);			 	//Prototype func for 6 ODEs: dP/dr, dM/dr, dv/dr and (3 for) dI/dr
__device__ static void rkdriver(double *Ppts, double *gamma0_param, double *acoef_param, double *rr, double *y_m); 	//Prototype func for Runge-Kutta integration driver function
#if __cplusplus
}
#endif

__constant__ double p_ns;		
__constant__ int numlinesSLY;					
__constant__ double rhopts[nparam+1]; 			  
__constant__ double p_SLY[lines+1], epsilon_SLY[lines+1], rho_SLY[lines+1];
//__constant__ double *P1, *P2, *P3, *P4, *P5;	

extern "C"
int kernel_driver(int myid, int n1overE, double *P1_1overE, double *P2_1overE, double *P3_1overE, double *P4_1overE, double *P5_1overE, double *rho_SLY_host, double *p_SLY_host, double *eps_SLY_host, int nSLY_host)
{

	int i;
	double *P1_1overE_d, *P2_1overE_d, *P3_1overE_d, *P4_1overE_d, *P5_1overE_d; 
	double *rhopts_host;
	double *rr_host, *mm_host;
	double *rr_dev, *mm_dev;
	double p_ns_host;

	rhopts_host = (double*)malloc((nparam+1)*sizeof(double));

	/*post_host = (double*)malloc((nMC+1)*sizeof(double));
	P1_host = (double*)malloc((nMC+1)*sizeof(double));
	P2_host = (double*)malloc((nMC+1)*sizeof(double));
	P3_host = (double*)malloc((nMC+1)*sizeof(double));
	P4_host = (double*)malloc((nMC+1)*sizeof(double));
	P5_host = (double*)malloc((nMC+1)*sizeof(double));

	char infile[256];
	i=1;
	//for (j=102; j<=102; j++)
	for (j=0; j <= nFiles; j++)
	{
		sprintf(infile,"chain_output/inversion_output_%d.txt",j);
		file = fopen(infile, "rt");
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

	max_P_host = max_array(post_host,nMC);
	oneOverE = max_P_host * exp(-1.);
	//oneOverSqrtE = max_P * exp(-0.5);

	int n1overE = 0;
	for (i=1; i<=nMC; i++)
		if (post_host[i] > oneOverE) n1overE++;

	post_1overE = (double*)malloc((n1overE+1)*sizeof(double));
	P1_1overE = (double*)malloc((n1overE+1)*sizeof(double));
	P2_1overE = (double*)malloc((n1overE+1)*sizeof(double));
	P3_1overE = (double*)malloc((n1overE+1)*sizeof(double));
	P4_1overE = (double*)malloc((n1overE+1)*sizeof(double));
	P5_1overE = (double*)malloc((n1overE+1)*sizeof(double));
	*/

	gpuErrchk( cudaMalloc(&P1_1overE_d, (n1overE+1)*sizeof(double)) );
	gpuErrchk( cudaMalloc(&P2_1overE_d, (n1overE+1)*sizeof(double)) );
	gpuErrchk( cudaMalloc(&P3_1overE_d, (n1overE+1)*sizeof(double)) );
	gpuErrchk( cudaMalloc(&P4_1overE_d, (n1overE+1)*sizeof(double)) );
	gpuErrchk( cudaMalloc(&P5_1overE_d, (n1overE+1)*sizeof(double)) );

	rr_host = (double*)malloc((n1overE*nEpsilon+1)*sizeof(double)); 
	mm_host = (double*)malloc((n1overE*nEpsilon+1)*sizeof(double));
	gpuErrchk( cudaMalloc(&rr_dev, (n1overE*nEpsilon+1)*sizeof(double)));
	gpuErrchk( cudaMalloc(&mm_dev, (n1overE*nEpsilon+1)*sizeof(double)));

	/*j=1;
	for (i=1; i<=nMC; i++)
	{
		if (post_host[i] > oneOverE)
		{
			post_1overE[j] = post_host[i];
			P1_1overE[j] = P1_host[i];
			P2_1overE[j] = P2_host[i];
			P3_1overE[j] = P3_host[i];
			P4_1overE[j] = P4_host[i];
			P5_1overE[j] = P5_host[i];
			j++;
		}
	}*/

	gpuErrchk( cudaMemcpy(P1_1overE_d, P1_1overE, (n1overE+1)*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpy(P2_1overE_d, P2_1overE, (n1overE+1)*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpy(P3_1overE_d, P3_1overE, (n1overE+1)*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpy(P4_1overE_d, P4_1overE, (n1overE+1)*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpy(P5_1overE_d, P5_1overE, (n1overE+1)*sizeof(double), cudaMemcpyHostToDevice));
	

	//readinSLY(p_SLY_host, eps_SLY_host, rho_SLY_host, &nSLY_host);
	getRhoPts(rhopts_host);
	p_ns_host = bisect_linint(rho_ns, rho_SLY_host, p_SLY_host, nSLY_host);				//pressure at rho_saturation according to sly 

	gpuErrchk( cudaMemcpyToSymbol(p_ns, &p_ns_host, sizeof(double)));
	gpuErrchk( cudaMemcpyToSymbol(numlinesSLY, &nSLY_host, sizeof(int)));	

	gpuErrchk( cudaMemcpyToSymbol(rhopts, rhopts_host, (nparam+1)*sizeof(double)));
	gpuErrchk( cudaMemcpyToSymbol(epsilon_SLY, eps_SLY_host, (lines+1)*sizeof(double)));	
	gpuErrchk( cudaMemcpyToSymbol(rho_SLY, rho_SLY_host, (lines+1)*sizeof(double)));	
	gpuErrchk( cudaMemcpyToSymbol(p_SLY, p_SLY_host, (lines+1)*sizeof(double)));	

	char filename[256];
	sprintf(filename, "MR_output/MR_oneOverE_%d.txt", myid);
	FILE *f_oneOverE = fopen(filename,"w");
	fprintf(f_oneOverE, "R       M    \n");

	/*char filename2[260];
	sprintf(filename2, "MR_output/MR_oneOverSqrtE_%d.txt",myid);
	FILE *f_oneOverSqrtE = fopen(filename2,"w");*/
	
	fprintf(f_oneOverE, "n 1/E for this process: %d \n", n1overE);
	fprintf(f_oneOverE, "num threads per block: 256  n blocks: %d \n", n1overE / 256);
	fprintf(f_oneOverE, "expected number of MR pts: %d \n", n1overE*nEpsilon);
	fflush(f_oneOverE);

	getMR_fromPs<<< (n1overE / 256), 256 >>>(n1overE, rr_dev, mm_dev, P1_1overE_d, P2_1overE_d, P3_1overE_d, P4_1overE_d, P5_1overE_d);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	gpuErrchk( cudaMemcpy(rr_host, rr_dev, (n1overE*nEpsilon+1)*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy(mm_host, mm_dev, (n1overE*nEpsilon+1)*sizeof(double), cudaMemcpyDeviceToHost));

	for (i=1; i<=n1overE*nEpsilon; i++)
		fprintf(f_oneOverE, "%f  %f \n", rr_host[i], mm_host[i]); 

	fclose(f_oneOverE);
	//fclose(f_oneOverSqrtE);

	
	gpuErrchk( cudaFree(rr_dev));
	gpuErrchk( cudaFree(mm_dev));
	gpuErrchk( cudaFree(P1_1overE_d));
	gpuErrchk( cudaFree(P2_1overE_d));
	gpuErrchk( cudaFree(P3_1overE_d));
	gpuErrchk( cudaFree(P4_1overE_d));
	gpuErrchk( cudaFree(P5_1overE_d));
	
	free(rhopts_host);
	free(rr_host);
	free(mm_host);

	return 0;
}

extern "C"
__global__ void getMR_fromPs(int nn, double *rr_super, double *mm_super, double *P1, double *P2, double *P3, double *P4, double *P5 )
{
	double *Ppts;
	double *rr_indiv, *mm_indiv;
	int i, j, index;


	Ppts = (double*)malloc((nparam+1)*sizeof(double));
	rr_indiv = (double*)malloc((nEpsilon+1)*sizeof(double));
	mm_indiv = (double*)malloc((nEpsilon+1)*sizeof(double));

	
	i = blockIdx.x*blockDim.x + threadIdx.x + 1;
	index = (i-1)*nEpsilon;

	if (i <= nn)
	{
		Ppts[1] = p_ns;
		Ppts[2] = P1[i];
		Ppts[3] = P2[i];
		Ppts[4] = P3[i];
		Ppts[5] = P4[i];
		Ppts[6] = P5[i];

		tov_dev(rr_indiv, mm_indiv, Ppts);

		for (j=1; j<=nEpsilon; j++) 
		{
			rr_super[index+j] = rr_indiv[j]; 
			mm_super[index+j] = mm_indiv[j];
		}

	}

	free(Ppts);
	free(rr_indiv);
	free(mm_indiv);
}

extern "C"
__device__ void tov_dev(double *rr, double *y_m, double *Ppts )
/* Main driver function to get M and  R (moment of inertia disabled)
   by solving the TOV equations (+ ODEs for I)
*/
{
	double *gamma0_param, *acoef_param;
	gamma0_param = (double*)malloc((nparam+1)*sizeof(double));	
	acoef_param = (double*)malloc((nparam+1)*sizeof(double)); 
	
	getGammaA(Ppts, gamma0_param,acoef_param);	//Find Gamma and a_coef along each segment, given Ppts
	rkdriver(Ppts, gamma0_param, acoef_param, rr, y_m);				//Call Runge-Kutta integrating function


	//for (j=1;j<=nEpsilon;j++)  inertia[j] = scaleI(j);			//Scale the trial moment of inertia, as in Kalogera & Psaltis 99

	free(gamma0_param);
	free(acoef_param);
}

extern "C"
__device__ void rkdriver(double *Ppts, double *gamma0_param, double *acoef_param, double *rr, double *y_m)
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
	__device__ void rk4(double y_eps, double y[],				//Prototype for 4th-order RK stepping function
		 double dydr[], double r, double h, double yout[],
		 double *yout_eps, double *Ppts, double *gamma0_param, double *acoef_param);
		
	int i,j,k;
	double r,hthisstep,h_Pstep,h_Mstep;
	double v_eps, vout_eps;
	double *v, *vout;
	double *dv;							//Order ALWAYS: M, P, v, f, cassi, I
	//double *initM, *initRho, *initEpsilon, *centralP;
	double initM, initRho, initEpsilon, centralP;
	
	v = (double*)malloc((nODEs+1)*sizeof(double));
	vout = (double*)malloc((nODEs+1)*sizeof(double));
	dv = (double*)malloc((nODEs+1)*sizeof(double));

	/*initRho = (double*)malloc((nEpsilon+1)*sizeof(double));
	initEpsilon = (double*)malloc((nEpsilon+1)*sizeof(double));
	initM = (double*)malloc((nEpsilon+1)*sizeof(double));
	centralP = (double*)malloc((nEpsilon+1)*sizeof(double));
	*/

	//initCond_dev(initM, initRho, initEpsilon, centralP, Ppts, gamma0_param, acoef_param);

	//y_v=dvector(1,nEpsilon);					//Assign memory for nu
	//y_f=dvector(1,nEpsilon);					//Assign memory for f
	//y_i=dvector(1,nEpsilon);					//Assign memory for I (unscaled)
	//centralP=dvector(1,nEpsilon);

	for (i=1; i<=nEpsilon; i++)					//Step through all values of rho_c
	{
	  //for (index=1; index<=2; index++)				//Loop through each rho_c twice, to determine starting value of nu
	  //{	
		initRho = eps_min*pow(1.07,i-1);
	
		if (initRho <= rhopts[1])
		{
			initEpsilon = findEps0_inSLY(initRho);
			centralP = EOSpressure(initRho, Ppts, gamma0_param, 0);
		} else
		{
			initEpsilon = param_massToEnergy(initRho, Ppts, gamma0_param, acoef_param);
			centralP = EOSpressure(initRho, Ppts, gamma0_param, 1);
		}
		initM = r_start*r_start*r_start*initEpsilon;


		r=r_start;							//First step located at initial radius
		v_eps=initEpsilon;           			//Load initial values for density, rho

		v[1]=initM;             				//Load initial values for M
		v[2]=centralP;


		/*if (index==1) 
			v[3]=1.0;					//Load initial value for nu at center = 0
		else
			v[3]=findnu(i);

		v[4]=1.0;						//Initial f at center -- of order unity, will be scaled!
		v[5]=0.0;						//Initial cassi = df/dr = 0 (at center)
		v[6]=0.0;						//Load initial mom of I
		*/
		k=1;	
		
		while (k <= nstep && v[2] >= Pedge*centralP)		//Keep stepping until P_edge (v_p) is negligible or exceed nstep	
		{
			h_Pstep= (v[2]*r*r) /						//Define stepsize from minimum of dP/dr 
				 ( (v_eps+v[2])*(v[1] + 3.0*r*r*r*v[2])*MoverR );  	// and dM/dr, calculated the the local environment    
			h_Mstep = v[1] / (3.0*r*r*v_eps);				// at each step
			hthisstep = fminf(fabsf(h_Pstep), fabsf(h_Mstep))*hscale;

			for (j=1;j<=nODEs;j++) dv[j]=derivs(Ppts, gamma0_param, acoef_param, j,r,v);      //,v[3],v[4],v[5]); 	//Get initial derivs for all 6 funcs
			rk4(v_eps, v, dv, r, hthisstep, vout, &vout_eps, Ppts, gamma0_param, acoef_param);
			r=r+hthisstep;	    					//Increase the radius increment		

		   	v_eps=vout_eps;	    					//Save updated terms
			v[1]=vout[1];	    					//mass				
			v[2]=vout[2];	    					//pressure	
			//v[3]=vout[3];	    					//nu			
			//v[4]=vout[4];	    					//f_trial	
			//v[5]=vout[5];	    					//cassi		
			//v[6]=vout[6];						//I_trial
			k+=1;							//Increase the stepper
			if (k-1==nstep && v[2] >= Pedge*centralP)		//Verify that we've actually reached edge w/in nsteps
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

	free(v);
	free(vout);
	free(dv);
	/*free(initRho);
	free(initEpsilon);
	free(initM);
	free(centralP);*/
}
extern "C"
__device__ void rk4(double y_eps, double y[], double dydr[], double r, double h, 
	double yout[], double *yout_eps, double *Ppts, double *gamma0_param, double *acoef_param)
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
	double *dym, *dyt;						//Order ALWAYS: M, P, v, f, cassi, I
	double *yt;

	dym = (double*)malloc((nODEs+1)*sizeof(double));
	dyt = (double*)malloc((nODEs+1)*sizeof(double));
	yt = (double*)malloc((nODEs+1)*sizeof(double));

	hh = h*0.5;              							//Half-step increment
    	h6 = h/6.0;             							//1/6th step increment
    	rh = r+hh;           								//Midpoint x-value
	
	for (j=1;j<=nODEs;j++) yt[j] = y[j] + hh*dydr[j];				//First step
	for (j=1;j<=nODEs;j++) dyt[j]=derivs(Ppts, gamma0_param, acoef_param, j,rh,yt);				 	//Second step
	for (j=1;j<=nODEs;j++) yt[j] = y[j] + hh*dyt[j];
	for (j=1;j<=nODEs;j++) dym[j]=derivs(Ppts, gamma0_param, acoef_param, j,rh,yt);					 //Third step			 						
	for (j=1;j<=nODEs;j++) 
	{
		yt[j] = y[j] + h*dym[j];
		dym[j] += dyt[j];
	}	
	for (j=1;j<=nODEs;j++) dyt[j]=derivs(Ppts, gamma0_param, acoef_param,j,r+h,yt);	//Fourth step
	for (j=1;j<=nODEs;j++) yout[j] = y[j]+h6*(dydr[j]+dyt[j]+2.0*dym[j]);		//Accumulate increments with proper weights
	if (yout[2] <= Ppts[1]) 							//Calculate final energy density at 3rd step from inverse EOS
		*yout_eps = EOSdensity(yout[2],Ppts, gamma0_param, acoef_param,0);					//Using SLY if in the low-density regime
	else
		*yout_eps = EOSdensity(yout[2],Ppts, gamma0_param, acoef_param,1);					//Otherwise use parametrized EOS		
	free(dym);
	free(dyt);
	free(yt);
}

extern "C"
__device__ double EOSpressure(double massdensity, double Ppts[], double gamma0_param[], int useparam)
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
		while (fabsf(b-a) > 1);						//until converged to within 2 tabulated values (index diff=1)

		m= (p_SLY[b] - p_SLY[a]) / (rho_SLY[b] - rho_SLY[a]);		 //Calculate slope to linearly interpolate btwn a and b
		intercept = p_SLY[a] - m*rho_SLY[a];				//Calculate y-intercept
		pressure = m*massdensity + intercept;				//get pressure

		if ( pressure > p_SLY[numlinesSLY] || pressure < p_SLY[1] )	//check results of bisection search
		{
			printf("Warning: bisection method found P %e at edge of range %d %d \n",pressure*p_char,a,b);
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

extern "C"
__device__ double EOSdensity(double pressure, double *Ppts, double *gamma0_param, double *acoef_param,int useparam)
/* Find the EoS density given a pressure, by searching through the tabulated
   values of pressure and density. Use bisection method to search.	
*/
{
	double massdensity, energydensity;
	int xmid;							//Index at midpoint
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
		} while (fabsf(b-a) > 1);					//until converged to within 2 tabulated values (index diff=1)

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

  
extern "C"
__device__ double derivs(double *Ppts, double *gamma0_param, double *acoef_param, int j, double r, double array[])  
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
 		density=EOSdensity(pressure,Ppts, gamma0_param, acoef_param, 0); 			//Using SLy if low-density
	else
		density=EOSdensity(pressure,Ppts, gamma0_param, acoef_param, 1); 			//And PARAMETRIZED EoS otherwise	

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

extern "C"
__device__ double param_massToEnergy(double massdensity, double Ppts[], double gamma0_param[], double acoef_param[])
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

	if (fabsf(gamma0_param[index]-1.0) < 0.01)				//If we're in a shallow segment (gamma=1), use special case of eps-rho conversion
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


extern "C"
__device__ double findEps0_inSLY(double rho0)
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
	while (fabsf(b-a) > 1);	

	m= (epsilon_SLY[b] - epsilon_SLY[a]) / (rho_SLY[b] - rho_SLY[a]);	//Calculate slope to linearly interpolate btwn a and b
	intercept = epsilon_SLY[a] - m*rho_SLY[a];				//Calculate y-intercept
	eps0 = m*rho0 + intercept;
	return eps0;
}


extern "C"
__device__ void getGammaA(double Ppts[], double *gamma0_param, double *acoef_param)
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
			if (fabsf(gamma0_param[j]-1.0) < 0.01)					//if gamma=1, use special case of eps-rho relation
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
			if ( fabsf(gamma0_param[j]-1.0) < 0.01)					//if gamma=1, use special case of eps-rho relation
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

extern "C"
__device__ void initCond_dev(double *initM, double *initRho, double *initEpsilon, double *centralP, double *Ppts, double *gamma0_param, double *acoef_param)
{

	int i,j;

	getGammaA(Ppts, gamma0_param, acoef_param);	       //Get gamma and 'a' coefs for this set of Ppts
	initRho[1] = 1.0*eps_min;
	for (i=2;i<=nEpsilon;i++) initRho[i] = initRho[i-1]*1.07;	//Scale starting mass density values
	for (j=1;j<=nEpsilon;j++)					//Convert init rho to eps
	{	
		if (initRho[j] <= rhopts[1])
		{
			initEpsilon[j] = findEps0_inSLY(initRho[j]);
			centralP[j] = EOSpressure(initRho[j],Ppts, gamma0_param, 0);
		}
		else	
		{		
			initEpsilon[j] = param_massToEnergy(initRho[j],Ppts,gamma0_param,acoef_param);
			centralP[j] = EOSpressure(initRho[j],Ppts,gamma0_param, 1);
		}
		initM[j] = r_start*r_start*r_start*initEpsilon[j];		//Initial masses enclosed by r_start
	}

}
