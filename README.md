# AI-Umpire

This repository comprises the work completed for my undergraduate dissertation. The AI Umpire system is a computer vision system that aims to assist Squash umpires in making in/out decisions. The system is composed of 4 phases:
1. Data Generation:
   * This phase involves the PyChrono simulation package interfacing with the POVRay raytracing renderer. 
   * PyChrono is used to somewhat realistically simulate a professional Squash match. This simulation data is then exported to POV Ray where it is rendered and saved to disk.
   * The simulation data (ball and player positions over time) is also saved to disk.
2. Video Generation:
   * In this phase, the rendered frames are converted into .MP4 videos with motion blur being added before encoding.
3. Ball Detection:
   * This phase uses the data generated in phases 1 and 2 and non-synthetic data can also be passed as input here.
   * The detector that I settled on is a contour detector but with multiple pre-processing steps to maximise detection performance, these are listed below:
     1. Segment foreground via frame differencing (using a sliding window of 3 frames).
     2. Blur image with a Gaussian filter to reduce noise.
     3. Binary threshold image to binarize and also reduce some noise.
     4. Apply the open morphological operator.
     5. Detect contours
4. Tracking:
   * Phase 3 outputs a candidate ball detection for each frame processed, due to unreliable detection, possible occlusions and to enable probabilistic interpretation of the trajectory. A Kalman filter is applied to track the ball over time and smooth out the noise in the measurements (candidate detections).
5. Trajectory Interpretation:
   * The Kalman filter stores a covariance matrix indicating its confidence in its prediction of the true ball position given the noisy measurement in the form of a covariance matrix.
   * This covariance matrix in combination with the predicted true position (the mean vector) is sampled in a regular, dense grid. 
   * These sample points represent possible positions of the ball each with a likelihood calculated with using the multivariate Gaussian density function.
   * Collision between each sample point and each wall and plane that defines an out-of-court region is computed giving a probability of the ball being out-of-court in that particular time-step when the out-of-court volumes only are considered.   

