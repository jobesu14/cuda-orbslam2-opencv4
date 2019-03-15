#include <map>
#include "functions.hpp"
#include "common.hpp"
#include <opencv2/core/core.hpp>
std::mutex mutex_data;

void ScaleCustomVector::print_size_input() {
	std::cout << in_kp->size_element() << std::endl;
}

void ScaleCustomVector::wait_output() {
	out_kp->waitTransfer();
}
void ScaleCustomVector::transfer_input() {
	in_kp->wait(where);
}
void ScaleCustomVector::transfer_output() {
	out_kp->setWhere(where);
	out_kp->finishProcessing();
}

void ScaleCustomVector::shouldTransferOutput( bool b ) {
	out_kp->setToBeTransferred( b );
}

MeasureReturn ScaleCustomVector::execute_gpu() {
	//todo launch exception?
	this->where = CPU;
	execute_cpu();
	return MeasureReturn(-1);
}

MeasureReturn ScaleCustomVector::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	out_kp->setSize(0);
	for (size_t i = 0; i < in_kp->size(); i++) {
		cv::KeyPoint kp = this->in_kp->vec[i];
		kp.pt.x *= scale;
		kp.pt.y *= scale;
		
		kp.size = scalePatch;
		
		out_kp->push_back(kp);
    }
    
    //std::cout << "IN_kp SCALEVECTOR " << level << ": " << in_kp->size() << std::endl;
    //std::cout << "OUT_kp SCALEVECTOR " << level << ": " << out_kp->size() << std::endl;
    
	return m;
}
