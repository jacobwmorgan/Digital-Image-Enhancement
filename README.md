# Digital Image Enhancement

This was a project for my final year computer science module , parallel programming.

```

Your task is to develop a digital image enhancement program that will perform contrast
adjustment using the histogram equalisation algorithm. The algorithm uses a cumulative
intensity histogram to back-project original image intensities resulting in an image of equalised
intensities. The algorithm is described in detail in [1], with the main calculation steps
summarised below and in Fig. 1:
- Calculate an intensity histogram from the input image (Fig. 1b).
- Calculate a cumulative histogram (Fig. 1c).
- Normalise and scale the cumulative histogram (Fig. 1d): the cumulative frequencies
are normalised and scaled to represent output image intensities (e.g. from 0-255 for an
8-bit image).
- Back-projection: the normalised cumulative histogram is used as a look-up table (LUT)
for mapping of the original intensities onto the output image. For each output pixel, the
algorithm should use the original intensity level as an index into the LUT and assign
the intensity value stored at that index.
- The output should be an intensity equalised image (Fig. 1e).

Due the large amount of data, all image processing must be performed on parallel hardware
and implemented by parallel software written in OpenCL with C++. You should develop your
own device code (i.e. kernels) that perform the main steps of the algorithm. The steps include
several classic parallel patterns including scan, histogram, and map. Your program should
also report memory transfer, kernel execution, and total program execution times for
performance assessment

```

