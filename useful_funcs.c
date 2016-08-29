/* C. Raithel, May 2016
 This MODULE contains generic functions that are useful
 in most programs 
 (e.g., a bisection function, min and max functions, etc.)
 
*/

#include "header_useful.h"

extern double mymin(double x1, double x2)
/* Return the smaller value of x1, x2 */
{
	if (x1 < x2)
		return x1;
	else 
		return x2;
}

extern double mymax(double x1, double x2)
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

extern int min_array_index(double array[], int num_elements)
/* Return the index of the smallest element in array */
{
   int i, index;
   double min=32000.;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i] < min)
	 {
	    min = array[i];
	    index = i;
	 }
   }
   return index;
}


extern int max_array_index(double array[], int num_elements)
/* Return the index of the largest element in array */
{
   int i, index;
   double max=-32000.;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i]>max)
	 {
	    max=array[i];
	    index = i;
	 }
   }
   return index;
}
extern int min_iarray(int array[], int num_elements)
/* Return the smallest element in an INT array */
{
   int i;
   int min=32000;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i] < min)
	    min = array[i];
	 
   }
   return min;
}


extern int max_iarray(int array[], int num_elements)
/* Return the largest element in an INT array */
{
   int i;
   int max=-32000;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i]>max)
	    max=array[i];
   }
   return max;
}

extern int min_iarray_index(int array[], int num_elements)
/* Return the smallest element in array */
{
   int i, index;
   int min=32000;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i] < min)
	 {
	    min = array[i];
	    index = i;
	 }
   }
   return index;
}


extern int max_iarray_index(int array[], int num_elements)
/* Return the largest element in array */
{
   int i, index;
   int max=-32000;
   for (i=1; i<=num_elements; i++)
   {
	 if (array[i]>max)
	 {
	    max=array[i];
	    index = i;
	 }
   }
   return index;
}
extern int bisection(double x, double x_array[],int numlines)
/* Use bisection method to search for the closest value to "x" 
   in the array "x_array".
*/
{
	int xmid;							//Index at midpoint
	int a=1,b=numlines, lower_index;				//Index bounds for whole interval
	do
	{
		xmid=(a+b)/2;
		if (x < x_array[xmid])
			b = xmid;
		else
			a = xmid;
	} 
	while (fabs(b-a) > 1);	

	lower_index = mymin(a,b);

	return lower_index;

}

extern double bisect_linint(double x, double x_array[],double y_array[], int lower_lim, int upper_lim)
/* Use bisection method to search for the value corresponding to "x" 
   in the array "y_array".
   Note: typically, lower_lim=1 and upper_lim=numlines, to search through entire array
*/
{
	int xmid;							//Index at midpoint
	int a=lower_lim, b=upper_lim, lower_index;				//Index bounds for whole interval
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

	if (x_array[a] != x_array[b])
	{
		m= (y_array[b] - y_array[a]) / (x_array[b] - x_array[a]);		//Calculate slope to linearly interpolate btwn a and b
		intercept = y_array[a] - m*x_array[a];				//Calculate y-intercept
		y = m*x + intercept;
	}
	else
		y = (y_array[a] + y_array[b])/2.;

	return y;
}

