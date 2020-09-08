This is project 1 of cs194-26 (Computational Photography Course at Berkeley)

# Images of the Russian Empire: Colorizing the Prokudin-Gorskii photo collection

## Project Overview

In 1907, Sergei Mikhailovich Prokudin-Gorskii (1863-1944) [Сергей Михайлович Прокудин-Горский] took color photographs of everything he saw. He accomplished this by using RGB filters and obtaining 3 glass plate negatives, RGB respectively. To view the colored images, we need to align and overlap the three plates. To improve results, it is also a good idea to make adjustments such as cropping, contrasting, white balance, color mapping, etc. The process of reproducing the color image is to be automated in this project.

## Project Approach

### Alignment

In this project, we are given input files with channels in the order BGR (as opposed to the conventional RGB). For simplicity, I aligned G and R channels to the B channel. My general approach is to have a fixed channel size of height _h_ and width _w_ . Then, we use an algorithm to find the best coordinate for the left-upper corner of each channel to which I call _start_ .

<emphasis>1\. Basic Algorithm</emphasis>

The first algorithm I adopted is to simply divide the image by three. So the _start_ for BGR would be (0, 0), (h, 0) and (h*2, 0). Then stace them together using numpy.stack. This method gave pretty reliable results with low computation time.

<emphasis>2\. Better Algorithms</emphasis>

I implemented both Sum of Squared Differences (SSD) and normalized cross-correlation (NCC) algorithms. I select a window that is smaller than the channel size to create sub-matrices for G and R channels. The sub-matrices are initialized to either 0 or infinity to suit the metric. This also help with handling out of bound windows.

## After Thoughts

### Difficulties

Buliding from scratch is really fun, but the downside it that it takes time to learn about new Python packages and to present the project as a whole nicely. Since there are so many parameters, I struggled to come up with a set of standardized arguments to pass into my alignment algorithms. Initially I wanted everything to be adjustable, but it made it really hard to test results as well as the debugging followed. This contributed to not being able to align the channels as good as I wanted.

### Wish List

I would like to implement adjustments before and after alignment. Maybe having a few repeated iterations of adjust, align, re-adjust, re-align can help find the best displacement values. I would also like to experiment more on adjustments that can be made to make the results look better.