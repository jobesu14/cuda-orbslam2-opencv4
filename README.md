# Dataflow ORB-SLAM2
**Authors:** Stefano Aldegheri, based upon [Raul Mur-Artal](http://webdiis.unizar.es/~raulmur/) ([ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2))

This implementation aims to target better efficiency in the feature extraction part using a dataflow description of the algorithm and using pipelining.
These enhancement ensure a real-time implementation in the embedded Jetson TX2 board, previously unachievable.

At the moment, only the monocular version of the KITTI dataset is supported.
