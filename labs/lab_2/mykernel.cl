/* Kernel Function for calculating Pi/4 */

__kernel void calculatePi(__global float *difference_array)
{
  /* Get the work group number */
  int work_group = get_global_id (0);

  float n = (float)work_group + 1.0f; 

  /* Calculate the part of the equation that corresponds to this work group */
  difference_array[work_group] = (1.0f/(4.0f*n-3.0f)) - (1.0f/(4.0f*n-1.0f));
}
