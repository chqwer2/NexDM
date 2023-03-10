# NexDM

### Step 1
Download image data with Depth (RGB-D), e.g. NYU dataset with Dense Depth Info, as Geometry Info

NYU-v2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html \\
RGB and Depth cameras from the Microsoft Kinect

[Labeled dataset (~2.8 GB)](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)

[Raw dataset, Multi-part (~428 GB)](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts)

[Toolbox](http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip)


### Step 2
Run MPI model forward: Split single-vire data into n layers, use the first layer

rendering 3D model from MPI model.
### Step 3
When render novel views, it will be holes and blocked parts, output Mask
### Step 4
Using the above info for Diffusion Model for Impainting
### Step 5
Train Nex



Notations

MPI: multiplane image


ref to papers: 

[1] Plug-and-play Diffusion Features for Text-Driven Image-to-Image Translation

[2] DreamFusion: Text-to-3D Using 2D Diffusion

[3] Nex: Real-time View Synthesis with Neural Basis Expansion

