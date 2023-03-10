#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		string userCommand;
		int bin_count;
		std::cout << "Enter number of bins | 256 for 8-bit" << std::endl;
		
		while (true)
		{
			getline(std::cin, userCommand);
			if (userCommand == "")
			{
				std::cout << "PLease enter a number " << std::endl;
				continue;
			}
			try { bin_count = std::stoi(userCommand); }
			catch (...) { std::cout << "Please enter a number" << std::endl; continue; }
		
			if (bin_count >= 0 && bin_count <= 256) { break; }
			else { std::cout << "Input a number between 0-256" << std::endl; continue; }
		}











		
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");
		const int IMAGE_SIZE = image_input.size();

		std::vector<int> Histogram(bin_count); //histogram, set to size of number of bins

		size_t local_size = Histogram.size(); //



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

		std::vector<int> int_histogram_buffer(Histogram.size());
		size_t int_hist_size = int_histogram_buffer.size() * sizeof(int);

		//Device Buffers
		cl::Buffer device_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer device_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		
		cl::Buffer int_histogram(context, CL_MEM_READ_WRITE, int_hist_size* sizeof(int));

		cl::Event event_one;
		cl::Event event_two;
		cl::Event event_three;
		cl::Event event_four;

		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &event_one);
		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &event_two);
		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &event_three);
		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &event_four);



		










	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
