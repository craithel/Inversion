/* C. Raithel, May 2016
 This MODULE contains generic functions that are useful
 in most programs 
 (e.g., a bisection function, min and max functions, etc.)
 
*/

#include "header_useful.h"

extern double min(double x1, double x2)
/* Return the smaller value of x1, x2 */
{
	if (x1 < x2)
		return x1;
	else 
		return x2;
}

extern double max(double x1, double x2)
/* Return the larger value of x1, x2 */
{
	if (x1 > x2)
		return x1;
	else 
		return x2;
}


extern double min_array(double array[], int num_elements)
/* Return the smallest element in array */
{
   int i;
   double min=32000.;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i] < min)
	 {
	    min = array[i];
	 }
   }
   return min;
}


extern double max_array(double array[], int num_elements)
/* Return the largest element in array */
{
   int i;
   double max=-32000.;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i]>max)
	 {
	    max=array[i];
	 }
   }
   return max;
}

extern int bisection(double x, double x_array[],int numlines)
/* Use bisection method to search for the value corresponding to "x" 
   in the array "y_array".
*/
{
	int xmid;							//Index at midpoint
	int a=1,b=numlines, lower_index;				//Index bounds for whole interval
	double m,intercept,y;						//Slope and int for linear interpolation

	do
	{
		xmid=(a+b)/2;
		if (x < x_array[xmid])
			b = xmid;
		else
			a = xmid;
	} 
	while (fabs(b-a) > 1);	

	lower_index = min(a,b);

	return lower_index;

}

