__kernel
void vecadd( __global char* img, __global int* background)
{
   unsigned int x = get_global_id(0);
   background[x*20] = x;
   img[x] = img[x] + 10;

}
