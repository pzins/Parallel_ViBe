__kernel
void vecadd( __global unsigned char* img, __global unsigned char* background,  __global int* randoms, __global unsigned char* output)//, __global unsigned int* debug)
{
   unsigned long long int x = get_global_id(0);
   int nb_matchs = 0;
   int dist;
   for(int j = 0; j < 20; j++)
   {
        dist = (background[x * 3 * 20 + j * 3] - img[x * 3]) * (background[x * 3 * 20 + j * 3] - img[x * 3]) +
               (background[x * 3 * 20 + j * 3 + 1] - img[x * 3 + 1]) * (background[x * 3 * 20 + j * 3 + 1] - img[x * 3 + 1]) +
               (background[x * 3 * 20 + j * 3 + 2] - img[x * 3 + 2]) * (background[x * 3 * 20 + j * 3 + 2] - img[x * 3 + 2]);
        if(dist <= 20*40)
            nb_matchs++;

    }
   if(nb_matchs >= 1) {
       if(randoms[0] == 0) {
            background[x * 3 * 20 + 3 * randoms[1]] = img[x * 3];
            background[x * 3 * 20 + 3 * randoms[1] + 1] = img[ x * 3 + 1];
            background[x * 3 * 20 + 3 * randoms[1] + 2] = img[ x * 3 + 2];
        }
        output[x] = 0;
    }
    else {
        output[x] = 255;
    }
}
