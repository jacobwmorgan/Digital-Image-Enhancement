//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) 
{
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void int_hist(global const uchar* A, global int* B)
{
	int id = get_global_id(0);
	int bin_index = A[id];
	atomic_inc(&B[bin_index]);
}