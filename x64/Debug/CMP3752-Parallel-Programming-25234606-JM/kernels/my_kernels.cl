//a simple OpenCL kernel which copies all pixels from A to B
typedef  unsigned short imageT;

kernel void identity(global const imageT* A, global imageT* B)
{
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void int_hist(global const imageT* A, global int* B, double binsize)
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


kernel void inclusive_hillis_steele(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;
	
	for (int stride = 1; stride<N; stride*= 2)
	{
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE);

		C = A; A = B; B = C;
	}
	B[id] = A[id];
	barrier(CLK_GLOBAL_MEM_FENCE);
	C = A; A = B; B = C;
}

kernel void blelloch_scan(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0); int t;
	
	for (int stride = 1; stride < N; stride *= 2)
	{
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (id == 0)A[N - 1] = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int stride = N / 2; stride > 0; stride /= 2)
	{
		if (((id + 1) % (stride * 2)) == 0)
		{
			t = A[id];
			A[id] += A[id - stride];
			A[id - stride] = t;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	B[id] = A[id];
}


kernel void hist_lut(global int* A, global int* B, int bin_count, int MAX_INTENSITY)
{
	int id = get_global_id(0);
	B[id] = A[id] * (double)MAX_INTENSITY / A[bin_count - 1];
}

kernel void back_proj(global imageT* A, global int* LUT, global imageT* B , double binsize) {
	int id = get_global_id(0);
	int group = A[id] / binsize;
	B[id] = LUT[group];
}