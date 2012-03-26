# A Stereo Approach that Handles the Matting Problem via Image Warping

This project is a end-course project for the **Advanced Methods For Computer Vision** course at ENS Cachan, under the supervision of **Nikos Paragios**.

The goal of the project was to re-implement and test the algorithm outlined in the paper _A Stereo Approach that Handles the Matting Problem via Image Warping_ by Michael Bleyer, Margrit Gelautz, Carsten Rother, and Christoph Rhemann.

In this file, I'll explain briefly the different stages of the implementation and the issues I encountered.

For those who need the [tl;dr](http://en.wikipedia.org/wiki/TLDR), the project was split in the following steps :

  1. Acquire stereo images
  2. Rectify them
  3. Compute the initial disparity map
  4. WarpMat
  
What you need in order to run this project :

  * A working **Python 2.x** distribution (I use 2.7)
  * The **numpy** and **matplotlib** packages (type `sudo pip install <package>` in a terminal to install a package in Python) for data manipulation and display
  * The **opencv** package and its Python bindings (usually installed with something like `sudo apt-get install opencv` in Ubuntu -- _not tested_)

## Rectifying Stereo Images

First, as in a lot of methods in stereo vision, we need rectified images. That is, we need the epipolar lines to coincide with the horizontal scanlines in both the left _and_ the right images.

![Scanlines in a rectified image pair](https://github.com/fxthomas/mva-advancedcv-project/raw/master/images/Rectify-Scanlines.png)

The `rectify(left, right)` method in `src/rectify.py` automatically rectifies a pair of stereo images in argument. It can be called in the terminal, like :

    src/$ python rectify.py left.png right.png
    src/$ ls
       left.png left-rectified.png right.png right-rectified.png

## Disparity map computation

The _WarpMat_ paper mentions the use of another algorithm for the disparity map computation, _Simple but effective tree structures for dynamic programming-based stereo matching_, by the same authors, Michael Bleyer and Margrit Gelautz, for the initial disparity map computation.

I implemented this algorithm as a Python module, `simpletree`.

The algorithm basically runs the same DP algorithm 8 times on the provided images, so the core DP method is written in C, for efficiency, and can be found in the `simpletree.dp` module.

![Disparity map for the Tsukuba image pair](https://github.com/fxthomas/mva-advancedcv-project/raw/master/images/Disparity-Tsukuba.png)

## Image segmentation

![Disparity map for the Tsukuba image pair](https://github.com/fxthomas/mva-advancedcv-project/raw/master/images/All-Subplots.png)
