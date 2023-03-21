#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;


typedef  unsigned short imageT;

CImgDisplay displayImage(CImg<imageT> im, bool is16Bit)
{
	if (is16Bit == false)
	{
		CImg<unsigned char> output_image = (CImg<imageT>) im;
		CImgDisplay disp_input(output_image, "image_8");
		return disp_input;
	}
	else{
		CImg<unsigned short> output_image = (CImg<imageT>) im;
		CImgDisplay disp_input(output_image, "image_16");
		return disp_input;
	};
}


void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}


int get_bins(bool is16bit)
{
	string userCommand;
	int bin_count;
	//Menu for asking how many bins 
		 // This is the amount of bins Il be using for the standard histogram, this will effect the amount of colours etc in the output image, as normalisation takes place.
	int requiredInput;
	if (is16bit) { requiredInput = 65536; }
	else { requiredInput = 256; };
	std::cout << "Enter number of bins | Maximum "<<requiredInput << std::endl;
	while (true)
	{
		getline(std::cin, userCommand);
		if (userCommand == "")
		{
			std::cout << "Please enter a number " << std::endl;
			continue;
		}
		try { bin_count = std::stoi(userCommand); }
		catch (...) { std::cout << "Please enter a number" << std::endl; continue; }

		if (bin_count >= 0 && bin_count <= requiredInput) { break; }
		else {std::cout << "Input a number between 0-"<<requiredInput<<std::endl; continue;}
	}
	return bin_count;

}

int main(int argc, char** argv) {
	
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm"; //Image name that is being read in (Can be changed in the options)

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//---------------------------------------------------------------------------------


	//detect any potential exceptions
	try {

		CImg<imageT> image_input; 

		int MAX_INTENSITY;
		bool is16Bit;

		CImg<imageT> temp_image(image_filename.c_str()); //A temporary storage for the image , as if RGB is used , I will need this image later on to input into the changed image, so i can put the colour back into the image
		CImg<imageT> cb, cr; //These are the two channels for blue and read in the YCbCr image type

		if (temp_image.max() <= 255) // 2^8 -1
		{
			std::cout << "Image is 8-bit" << std::endl;
			is16Bit = false;
			MAX_INTENSITY = 255;
		}
		else if (temp_image.max() <= 65535) // 2^16 -1
		{
			std::cout << "Image is 16-bit" << std::endl;
			is16Bit = true;
			MAX_INTENSITY = 65535;
		}

		bool is_RGB = false; //This is to check later on if i need to put the colours back in or not
		if (temp_image.spectrum() == 1) // spectrum() Checks how many channels there are in the image (Greyscale has one)
		
		{
			//This doesnt really do anything but set the image we want to use as the temp image
			std::cout << "Image is Gray scale" << std::endl;
			image_input = temp_image;
			is_RGB = false;
		}
		else if (temp_image.spectrum() == 3) // (RGB has 3)
		{
			std::cout << "Image is RGB" << std::endl;
			is_RGB = true;

			CImg<imageT> Ycbcr_Image = temp_image.get_RGBtoYCbCr(); //Converts the RGB image into a YCbCr image. YCbCr is preferred as it is designed for digital images
			
			image_input = Ycbcr_Image.get_channel(0); //
			cb = Ycbcr_Image.get_channel(1);
			cr = Ycbcr_Image.get_channel(2);
		}


		int bin_count = get_bins(is16Bit);

		CImgDisplay disp_input = displayImage(image_input, is16Bit);

		const int IMAGE_SIZE = image_input.size();
		

		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Histogram Code
		typedef int type; //Creating my own type here I can track the size of it later on
		std::vector<type> Histogram(bin_count); //histogram, set to size of number of bins

		size_t local_size = Histogram.size() * sizeof(type); //Gets the size in bytes

		//Setting histogram bin size based the size of the image
		//Device Buffers
		cl::Buffer device_image_input(context, CL_MEM_READ_ONLY, image_input.size() * sizeof(imageT));
		cl::Buffer device_image_output(context, CL_MEM_READ_WRITE, image_input.size()*sizeof(imageT));

		//Histogram buffer
		cl::Buffer device_int_histogram(context, CL_MEM_READ_WRITE, local_size);
		cl::Buffer device_cumulative_histogram_output(context, CL_MEM_READ_WRITE, local_size);
		cl::Buffer device_LUT_output(context, CL_MEM_READ_WRITE, local_size);


		//Copy images to buffer
		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, image_input.size() * sizeof(image_input[0]), &image_input.data()[0]);

		//-------------------------------------------------------------------------------------------
		//Creates a frequency histogrm all pixel values 0-255

		double bin_size = (MAX_INTENSITY + 1) / bin_count; //Gets the size of the bins using MAX_INTENSITY which changes based on if the imags is 16 or 8 bit

		cl::Kernel kernel_simple_histogram = cl::Kernel(program, "int_hist"); //Creates a new instance of the "int_hist" kernel function
		kernel_simple_histogram.setArg(0, device_image_input); //Sets the first argument as the location of the image in memory
		kernel_simple_histogram.setArg(1, device_int_histogram);// Sets the second argument as the location of the histogram in memory
		kernel_simple_histogram.setArg(2, bin_size); // Sets the third argument as the size of bins
		
		cl::Event hist_event; // used for profiling the kernel function
		queue.enqueueNDRangeKernel(kernel_simple_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hist_event); //Queue the function
		queue.enqueueReadBuffer(device_int_histogram, CL_TRUE, 0, local_size, &Histogram[0]); //Read the output of the function to memoery

		std::cout << "Histogram done" << std::endl;

		//-------------------------------------------------------------------------------------------
		//Cumulative histogram 
		std::vector<type> CumHistogram(bin_count);												
		queue.enqueueFillBuffer(device_cumulative_histogram_output, 0, 0, local_size); //Creates buffer as the size of the histogram

		cl::Kernel kernel_cumulative_histogram = cl::Kernel(program, "inclusive_hillis_steele"); // Creates a new instance of the "cum_hist" kernel function
		kernel_cumulative_histogram.setArg(0, device_int_histogram); //inputing histogram 
		kernel_cumulative_histogram.setArg(1, device_cumulative_histogram_output); //output to the bufffer

		cl::Event cumulative_hist_event;

		queue.enqueueNDRangeKernel(kernel_cumulative_histogram, cl::NullRange, cl::NDRange(CumHistogram.size()), cl::NullRange, NULL, &cumulative_hist_event);
		queue.enqueueReadBuffer(device_cumulative_histogram_output, CL_TRUE, 0, local_size, &CumHistogram[0]);

		std::cout << "cum Histogram done" << std::endl;
		//------------------------------------------------------------------------------------------
		//Look up table 

		std::vector<type> LUT(bin_count);

		queue.enqueueFillBuffer(device_LUT_output,0,0,local_size);

		cl::Kernel kernel_LUT = cl::Kernel(program, "hist_lut");
		kernel_LUT.setArg(0, device_cumulative_histogram_output);
		kernel_LUT.setArg(1, device_LUT_output);
		kernel_LUT.setArg(2, bin_count);
		kernel_LUT.setArg(3, MAX_INTENSITY);

		cl::Event lut_event;

		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(local_size), cl::NullRange, NULL, &lut_event);
		queue.enqueueReadBuffer(device_LUT_output, CL_TRUE, 0, local_size , &LUT[0]);

		std::cout << "LUT DONE" << std::endl;

		//------------------------------------------------------------------------------------------
		//Back projection

		cl::Kernel kernel_back_projection = cl::Kernel(program, "back_proj");
		kernel_back_projection.setArg(0, device_image_input);
		kernel_back_projection.setArg(1, device_LUT_output);
		kernel_back_projection.setArg(2, device_image_output);
		kernel_back_projection.setArg(3, bin_size);

		cl::Event back_projection_event;

		vector<imageT> output_buffer(image_input.size()*sizeof(imageT));
		queue.enqueueNDRangeKernel(kernel_back_projection, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &back_projection_event);

		queue.enqueueReadBuffer(device_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		std::cout << "BACK DONE" << std::endl;
		//------------------------------------------------------------------------------------------
		//Outputs

		std::cout << std::endl;
		std::cout << "Histogram: " << Histogram << std::endl;
		std::cout << "Histogram kernel execution time [ns]: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Histogram memory transfer: " << GetFullProfilingInfo(hist_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		std::cout << "Cumulative Histogram: " << CumHistogram << std::endl;
		std::cout << "Cumulative Histogram kernel execution time [ns]: " << cumulative_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulative_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Cumulative Histogram memory transfer: " << GetFullProfilingInfo(cumulative_hist_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;


		std::cout << "Look-up table (LUT): " << LUT << std::endl;
		std::cout << "LUT kernel execution time [ns]: " << lut_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lut_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "LUT memory transfer: " << GetFullProfilingInfo(lut_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;


		std::cout << "Vector kernel execution time [ns]: " << back_projection_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - back_projection_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Vector memory transfer: " << GetFullProfilingInfo(back_projection_event, ProfilingResolution::PROF_US) << std::endl;
		std::cout << std::endl;
		std::cout << "Overall execution time [ns]:  " << back_projection_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() << std::endl;



		//Image output

		CImg<imageT> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//RGB Output handling
		if (is_RGB == true)
		{
			CImg<imageT> RGBImg = output_image.get_resize(temp_image.width(),temp_image.height(),temp_image.depth(), temp_image.spectrum());
			for (int x = 0; x < temp_image.width(); x++)
			{
				for (int y = 0; y < temp_image.height(); y++)
				{
					RGBImg(x, y, 0) = output_image(x, y);
					RGBImg(x, y, 1) = cb(x, y);
					RGBImg(x, y, 2) = cr(x, y);
				}
			}
			
			output_image = RGBImg.get_YCbCrtoRGB();
		}
		
		


		CImgDisplay disp_output = displayImage(output_image, is16Bit);

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC())
		{
			disp_input.wait(1);
			disp_input.wait(1);
		}
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
