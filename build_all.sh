#!/bin/bash

function mkdir2()
{
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
}

build_type=(normal custom_cpu custom_gpu)
custom_vx=(OFF ON ON)
custom_vx_gpu=(OFF OFF ON)

pipeline=(no_pipeline pipeline_nowait pipeline_wait)
use_pipeline=(OFF ON ON)
wait_pipeline=(OFF OFF ON)

sync_type=(async sync_buffer sync_full)
sync_values=(0 1 2)

mkdir -p builds
cd builds

for i in ${!build_type[@]}; do
	mkdir -p ${build_type[$i]}
	cd ${build_type[$i]}
	for j in ${!pipeline[@]}; do
		mkdir -p ${pipeline[$j]}
		cd ${pipeline[$j]}
		for k in ${!sync_type[@]}; do
			mkdir -p ${sync_type[$k]}
			cd ${sync_type[$k]}

			cmake -DCMAKE_BUILD_TYPE=Release \
				-DUSE_CUSTOM_VX=${custom_vx[$i]} -DUSE_CUSTOM_VX_GPU=${custom_vx_gpu[$i]} \
				-DUSE_PIPELINE=${use_pipeline[$j]} -DWAIT_PIPELINE=${wait_pipeline[$j]} \
				-DSYNC_TYPE=${sync_values[$k]} ../../../../
			make -j7
			
			cd ..
		done
		cd ..
	done
	cd ..
done
cd ..

