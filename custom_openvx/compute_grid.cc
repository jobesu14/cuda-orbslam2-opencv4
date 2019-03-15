#include <map>
#include "functions.hpp"
#include "common.hpp"

void ComputeGrid::print_size_input() {
	std::cout << src->size_element() << std::endl;
}

void ComputeGrid::wait_output() {
	dst->waitTransfer();
}
void ComputeGrid::transfer_input() {
	src->wait(where);
}
void ComputeGrid::transfer_output() {
	dst->setWhere(where);
	dst->finishProcessing();
}

MeasureReturn ComputeGrid::execute_gpu() {
	this->where = CPU;
	//todo launch exception?
	execute_cpu();
	return MeasureReturn(-1);
}
void ComputeGrid::shouldTransferOutput( bool b ) {
	dst->setToBeTransferred( b );
}

MeasureReturn ComputeGrid::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	
    float newfast_value = threshold;
    const float W = WINDOW_SIZE_C;
    
    const int minBorderX = min_x-3;
	const int minBorderY = minBorderX;
	const int maxBorderX = max_x+3;
	const int maxBorderY = max_y+3;
    
    const float width = (maxBorderX-minBorderX);
	const float height = (maxBorderY-minBorderY);

	const int nCols = width/W;
	const int nRows = height/W;
	const int wCell = ceil(width/nCols);
	const int hCell = ceil(height/nRows);
	
	const float limit_x = (width - 6) / wCell;
	const float limit_y = (height - 6) / hCell;
	
/*	if(level == 1) {
		std::cout << "## " << std::endl;
		std::cout << "minBorderX: " << minBorderX << std::endl;
		std::cout << "maxBorderX: " << maxBorderX << std::endl;
		std::cout << "minBorderY: " << minBorderY << std::endl;
		std::cout << "maxBorderY: " << maxBorderY << std::endl;
		std::cout << "wCell: " << wCell << std::endl;
		std::cout << "hCell: " << hCell << std::endl;
		std::cout << "nRows: " << nRows << std::endl;
		std::cout << "nCols: " << nCols << std::endl;
		std::cout << "width: " << width << std::endl;
		std::cout << "heigh: " << height << std::endl;
		std::cout << "-- " << std::endl;
	}*/

    {
        dst->setSize(0);
        
        int sz = src->size();
        
        std::map<std::pair<int, int>, bool> grid;
        std::map<std::pair<int, int>, std::vector<cv::KeyPoint>> gridPoints;
        cv::KeyPoint* kps = src->vec;
        
        // first step: read and add certain threshold
        for (int i = 0; i < sz; i++)
        {
			cv::KeyPoint kp = kps[i];
			if(kp.response < newfast_value) continue; //not corner in first metric

            //precalcola la cella
            //int nColumn = floor((kp.pt.x-minBorderX)/wCell);
            //int nRow    = floor((kp.pt.y-minBorderY)/hCell);
            int nColumn = floor((kp.pt.x-min_x)/wCell);
            int nRow    = floor((kp.pt.y-min_y)/hCell);

            bool b1 = nColumn < nCols;
            bool b2 = nRow < nRows;
            bool b3 = nRow >= 0;
            bool b4 = nColumn >= 0;
            bool b5 = kp.pt.x >= min_x;
            bool b6 = kp.pt.x < max_x;
            bool b7 = kp.pt.y >= min_y;
            bool b8 = kp.pt.y < max_y;
            bool b9 = nColumn < limit_x;
            bool b0 = nRow < limit_y;

            if(b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8 && b9 && b0)
            {
				//dst->push_back(kp);
                grid[std::pair<int, int>(nRow, nColumn)] = true;
                gridPoints[std::pair<int, int>(nRow, nColumn)].push_back(kp);
            }
        }

        // second step: add threshold if grid empty
        for (int i = 0; i < sz; i++)
        {
            cv::KeyPoint kp = kps[i];
            
            if(kp.response >= newfast_value) continue; //corner already processed in first metric

            //precalcola la cella
            //int nColumn = floor((kp.pt.x-minBorderX)/wCell);
            //int nRow    = floor((kp.pt.y-minBorderY)/hCell);
            int nColumn = floor((kp.pt.x-min_x)/wCell);
            int nRow    = floor((kp.pt.y-min_y)/hCell);

            //if there is points in grid, move on
            if(grid[std::pair<int, int>(nRow, nColumn)]) continue;

            bool b1 = nColumn < nCols;
            bool b2 = nRow < nRows;
            bool b3 = nRow >= 0;
            bool b4 = nColumn >= 0;
            bool b5 = kp.pt.x >= min_x;
            bool b6 = kp.pt.x < max_x;
            bool b7 = kp.pt.y >= min_y;
            bool b8 = kp.pt.y < max_y;
            bool b9 = nColumn < limit_x;
            bool b0 = nRow < limit_y;

            //tmpKp[currentIdx].x -= min_x;
            //tmpKp[currentIdx].y -= min_y;

            if(b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8 && b9 && b0)
            {
                //dst->push_back(kp);
                gridPoints[std::pair<int, int>(nRow, nColumn)].push_back(kp);
            }
        }
        
        for(int c = 0; c < limit_x; c++) {
			for(int r = 0; r < limit_y; r++) {
				for(auto kp : gridPoints[std::pair<int, int>(r, c)]) {
					dst->push_back(kp);
				}
			}
		}
    }
        
    return m;
}

