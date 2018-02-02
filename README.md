# carvana-pose
Pose estimation for carvana dataset

## Data
I took all the Carvana data (everything they distribute as test and train for their image masking challenge) and adapted it for pose estimation by splitting it as follows:

Training: 3943 examples of each pose  
Validation: 1315 examples of each pose  
Test: 1314 examples of each pose  

This corresponds to a 60/20/20 split between train/validate/test.

## Example data
There are 16 poses for each car, evenly sampled across 360 degrees of yaw.  

Here are some samples

### Yaw 0
![Example 1, yaw 0](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw01/0cdf5b5d0ce1_01.png)
![Example 2, yaw 0](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw01/0d53224da2b7_01.png)

### Yaw 5 (90 degrees)
![Example 1, yaw 5](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw05/4cce4dafa50c_05.png)
![Example 2, yaw 5](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw05/4f37105bc81a_05.png)

### Yaw 9 (180 degrees)
![Example 1, yaw 9](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw09/1cd4b21b7496_09.png)
![Example 2, yaw 9](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw09/1ee65a0b1542_09.png)

### Yaw 13 (270 degrees)
![Example 1, yaw 13](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw13/38582acaeb4c_13.png)
![Example 2, yaw 13](https://github.com/lambertwx/carvana-pose/blob/master/example-images/yaw13/66097a7f34e6_13.png)
