# Resolution enhancement

Using an extended video acquisition, we can to increase the resolution of a Living Optics camera hypercube.

## Examples

GIF of enhancement process

![Colour targets](./media/res_enhance.gif)

Comparison of preview frames, default hypercube and resolution enhanced hypercube.

![New York Print](./media/super.png)

## How it works

The Living Optics camera stack has a high resolution preview/ RGB arm and a low resolution snapshot hyperspectral imaging arm.
The hyperspectral arm sparsely samples a select number of points in the Field of View (FOV).
Capturing video with relative motion between the scene and the camera will allow the hyperspectral arm to uniquely sample more points
This motion can be

- Slight tilting/ panning/ rolling the camera whilst mounted on a tripod
- Moving the tripod up and down/ left and right
- Natural handheld vibrations of the camera
- Induced handheld motion of the camera (e.g. figure eight)
- If the scene is planar (e.g. in microscope configuration or inspecting a flat print), it is also acceptable to move the scene

The locations of the uniquely sampled points are estimated using the Preview camera, which are then used to upsample the hypercube to a higher resolution

### Notes

This current implementation assumes that the scene is Flat/ planar or Very far away. The implementation will not handle parallax correctly, nor dynamic scenes
An acquisition with severe motion blur will result in incorrect hypercube estimation.

## How to do it

Acquisition

- Start recording datacube as normal
- Select exposure time/ frame rate as balance between
    - Data volume desired or required.
    - Magnitude of motion. Uncontrolled large motions e.g. jitter may require lower exposure times
    - Brightness of scene and the desired signal to noise ratio
    - Gain (if used)
- Some sample acquisition settings are
    - 200 ms exposure time / 5 frames per second (fps) for smooth panning motion on a tripod
    - 100 ms exposure time/ 10 fps for controlled handheld motion
    - 33 ms exposure time/ 30 fps for natural handheld vibrations


Before recording data, ensure that there is no movement in the first frame (either by ensuring the camera is still or the scene is still)
Start recording
Induce relative motion between the scene and the camera
Stop recording


### Notes

The current implementation assumes that all frames have a certain degree of spatial overlap with the initial frame
The algorithm will fail if one or more frames lie completely outside the initial FOV
To ensure this, always ensure at least 50% of the first frame is visible when inducing motion
This means to not pan/ tilt/ roll the camera to extreme amounts


## Usage

We can either import the modules in the Jupyter notebook, or use the function as a command line interface.

Running the Command line interface: `python3 run_res_enhance.py`.

```text
usage: Living Optics Resolution Enhancement CLI [-h] [--file FILE] [--calibration CALIBRATION] [--filter_snap FILTER_SNAP] [--downsampling_factor [DOWNSAMPLING_FACTOR]] [--mode [MODE]]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Path to loraw datafile.
  --calibration CALIBRATION
                        Path to calibration folder.
  --filter_snap FILTER_SNAP
                        Path to 600 nm filter snap file.
  --downsampling_factor [DOWNSAMPLING_FACTOR]
                        Integer downsampling factor.
  --mode [MODE]         Backend to run resolution enhancement in. Either 'homography' or 'phase_correlation'. Defaults to 'homography'

Living Optics 2024

```
