# DRAGNet
Rachael and Josh code

__Metadata__

Contents: 

•	LO1_calibration_files (Factory calibration folder which is needed for extracting the metadata, we use the 8.5mm folder)

•	Meta_Data.csv (csv where all the metadata is saved to)

•	Metadata_extract.ipynb (Python notebook which saves all image metadata to csv)

Metadata_extract.ipynb: 

1.	Must have sdk installed for the lo.sdk modules to run

2.	Alter directory for factory calibration if needed (shouldn’t need to if on desktop)


__GUI__

Contents:

•   pycache (don’t need to worry about)

•	LO1_calibration_files (Factory calibration needed to decode loraw files, we use 8.5mm folder)

•	GUI_Code.ipynb (Python notebook which runs the GUI)

•	metadata_gui.csv (csv file where metadata found in the GUI can be saved to) (All data previously put in this file will be overwritten unless you change the code)

•	res_enhance.py (Code created by Kenton which performs resultion enhancement on any pictures loaded into the GUI)

GUI_Code.ipynb:

1.	Need sdk downloaded for modules to all run

2.	GUI can be used to easily and quickly view loraw files and see what they look like enhanced as well as their metadata 

3.	Must click on the show image button before enhancing an image 

4.	Must close GUI when finished instead of stopping the code as this will restart the kernel

__Resolution_enhancement__:

Contents:

•	Josh (Joshs folder with the code for enhancing images)

•	Kenton (kentons untouched code for running resolution enhancement, this was just incase I edited it and messed it up so I could have a backup)

Inside ‘Josh’:

•   pycache (don’t need)

•	Envi_file_opener.ipynb (Used to open enhanced hypercube files and save a single wavelength as a png for easy viewing)

•	Res_enhance.py (copy of one of kentons python scripts with a few adjustments made near the end)

•	Res_enhancer_all.ipynb (Script to run the resolution enhancement code on all the loraw data and then save the hypercubes in an envi file and a numpy file in a folder called ‘Enhanced_images’ in LaCie)

Res_enhancer_all.ipynb:

1.	Need sdk to run all the modules

2.	Alter directory to Wytham data if needed 

3.	Alter directory to factory calibration folder if needed (I use the one inside the metadata folder)

4.	Notice that when running this code, the output on visual studio gets very large and can cause it to crash, therefore I would run it for one treatment in a certain block and then interrupt the code and pop this treatment from ‘data_folders’. That way it wouldn’t go over it again when I re run the code for the next treatment. 

5.	Res enhancement will be run for all files regardless if they are single pictures or videos (however it will only actually enhance resolution for the videos as this is required by the LO camera)

6.	Code saves 4 items to the folder ‘Enhanced_images’ inside LaCie at ‘LaCie/Living_Optics_hyperspec_Josh/Wytham_raw_data/Enhanced_images’. It saves a png file of the enhancement process, a numpy (.npy) file which saves the hypercube data as a 3d array, a .hdr and .img file which together form the envi file which also has the hypercube saved.

Envi_file_opener.ipynb:

1.	File and header_file variables are used for opening select envi files to see if they actually contain the correct data

2.	Can alter directory to correct location of all enhanced hypercubes

3.	Code will run through all enhanced hypercubes saved and will plot the npy file as a png using the last wavelength (95) which usually makes it the brightest

4.	Pngs will all be saved in a separate folder called ‘Spectral_images’

5.	Last two boxes of notebook are used to open select files in envi format or npy format to see if they are correct 

__Distortion_Correction__:

Contents:

•	Folder called ‘Images’. This folder houses all the images we want to distort and also their saved result. Any saved result will be overwritten if you decide to rerun the code and distort them in a different way.

•	‘Manual_distortion_code.ipynb’ – Python notebook which will run the distortion code (click run all when running it). 
Manual_distortion_code.ipynb:

1.	To start, click run all

2.	The code will pop up the undistorted images allowing the user to select the four corners of the quadrat in that image. Once four points have been selected with the mouse, a new window will pop up showing the resulting distorted image. Press the right arrow key to then progress to the next picture, press the left arrow key to go back a picture, and press the esc key to quit. If you want to alter the distorted image of only one picture, you can either remove all images from the ‘Images’ folder except the one of interest, or, you can press the right arrow key until you get to the image, distort it and then press esc.
