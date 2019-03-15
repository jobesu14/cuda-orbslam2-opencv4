#pragma once
#include <mutex>
#include <condition_variable>
#include <iostream>
#include "data_types.hpp"
extern Collector c;

class Semaphore {
public:
    Semaphore (int count_ = 0)
        : count(count_) {}

    inline void notify(std::string s = "")
    {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        cv.notify_one();
        if(!s.empty()) {
			//std::cerr << s << count << std::endl;
			c.push_back(s + std::to_string(count));
		}
    }

    inline void wait(std::string s = "")
    {
        std::unique_lock<std::mutex> lock(mtx);

        while(count == 0){
            cv.wait(lock);
        }
        count--;
        
        if(!s.empty()) {
			//std::cerr << s << count << std::endl;
			c.push_back(s + std::to_string(count));
		}
    }
    
    int num() {
		return count;
	}

private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};


class SemaphoreMax {
public:
    SemaphoreMax (int maxCount_ = 1, int count_ = 0)
        : maxCount(maxCount_), count(count_) {}

    inline void notify(std::string s = "")
    {
        std::unique_lock<std::mutex> lock(mtx);
        if(!(count == maxCount)) {
			// return;
			count++;
		}
        cv.notify_one();
        if(!s.empty()) {
			//std::cerr << s << count << std::endl;
			c.push_back(s + std::to_string(count));
		}
    }

    inline void wait(std::string s = "")
    {
        std::unique_lock<std::mutex> lock(mtx);

        while(count == 0){
            cv.wait(lock);
        }
        count--;
        
        if(!s.empty()) {
			//std::cerr << s << count << std::endl;
			c.push_back(s + std::to_string(count));
		}
    }
    
    int num() {
		return count;
	}

private:
    std::mutex mtx;
    std::condition_variable cv;
    int maxCount;
    int count;
};
