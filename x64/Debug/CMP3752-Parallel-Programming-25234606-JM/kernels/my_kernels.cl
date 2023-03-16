//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B)
{
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void int_hist(global const uchar* A, global int* B, double binsize)
{
	int id = get_global_id(0);
	int bin_index = A[id] / binsize;
	atomic_inc(&B[bin_index]);
}

kernel void cum_hist(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

kernel void hist_lut(global int* A, global int* B, int bin_count)
{
	int id = get_global_id(0);
	B[id] = A[id] * (double)255 / A[bin_count - 1];
}

kernel void back_proj(global uchar* A, global int* LUT, global uchar* B , double binsize) {
	int id = get_global_id(0);
	int group = A[id] / binsize;
	B[id] = LUT[group];
}