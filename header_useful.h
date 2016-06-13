#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

extern double mymin(double x1, double x2);
extern double mymax(double x1, double x2);
extern double min_array(double array[], int num_elements);
extern double max_array(double array[], int num_elements);
extern int min_array_index(double array[], int num_elements);
extern int max_array_index(double array[], int num_elements);
extern int min_iarray_index(int array[], int num_elements);
extern int max_iarray_index(int array[], int num_elements);
extern int min_iarray(int array[], int num_elements);
extern int max_iarray(int array[], int num_elements);
extern int bisection(double x, double x_array[],int numlines);
extern double bisect_linint(double x, double x_array[],double y_array[], int numlines);

#ifdef __cplusplus
}
#endif
