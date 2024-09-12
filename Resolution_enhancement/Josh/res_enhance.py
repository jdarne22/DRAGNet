import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as sdkopen
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm


class Densifier:
    """Class to handle conversion of LO spectral points ordered in (y,x) to a dense
    hypercube.
    """

    def __init__(
        self,
        interp_start_w: float,
        interp_end_w: float,
        interp_start_h: float,
        interp_end_h: float,
        sampling_coordinates: np.ndarray,
        coordinates_are_yx: bool = True,
    ):
        """Initialises fixed square range for interpolation.

        Args:
            interp_start_w (float): Starting coordinate (width) for interpolation.
            interp_end_w (float): Ending coordinate (width) for interpolation.
            interp_start_h (float): Starting coordinate (height) for interpolation.
            interp_end_h (float): Ending coordinate (height) for interpolation.
            sampling_coordinates (np.ndarray): Fixed coordinates where spectra is
                sampled.
            coordinates_are_yx (bool): If sampling coordinates provided are ordered in
                (y,x) instead of (x,y).
        """
        self.interp_start_w = interp_start_w
        self.interp_end_w = interp_end_w
        self.interp_start_h = interp_start_h
        self.interp_end_h = interp_end_h

        self.coords = sampling_coordinates
        # Change coordinate ordering from (y,x) to (x,y) to agree with mesh grid
        # ordering.
        if coordinates_are_yx:
            self.coords = self.coords[:, ::-1]

        self.X_eval, self.Y_eval = np.meshgrid(
            np.arange(self.interp_start_w, self.interp_end_w),
            np.arange(self.interp_start_h, self.interp_end_h),
        )

    def __call__(self, z_sample):
        """Converts sampled points to a cube using the
            NearestNDInterpolator. Sparse cube is sampled at
            points specified by self.coords and will be
            interpolated using the meshgrid given in the init.

        Args:
            z_sample (np.ndarray): array shape (N, C)
            where N = len(sampling_coordinates) and C is the number of channels

        Returns:
            result: cube interpolated
        """
        interp = NearestNDInterpolator(self.coords, z_sample)
        res = interp(self.X_eval, self.Y_eval)
        return np.asarray(res)


class MultiViewGeometry:
    def __init__(self):
        self.kp_extractor = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

        self.homography_matrix = None

    def homography(self, im1, im2):
        """Calculates the homography between two images

        Args:
            im1 (np.ndarray[np.uint8]): Source Image, A 3D array of with shape (H,W,3)
            im2 (np.ndarray[np.uint8]): Destination Image, A 3D array of with shape
            (H,W,3)

        Raises:
            AssertionError: Raised when number of keypoints
            between two images are less than 5

        Returns:
            homography (np.ndarray): Homography, A 2D array of with shape 3x3
        """
        kp1, des1 = self.kp_extractor.detectAndCompute(im1, None)
        kp2, des2 = self.kp_extractor.detectAndCompute(im2, None)

        if len(kp1) < 8:
            logging.warning("Not enough keypoints found! Check quality of scene frame.")
            self.homography_matrix = None
            return None

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
        matches = np.asarray(good)

        if len(matches.shape) > 1 and len(matches[:, 0]) >= 5:
            src = (
                np.asarray([kp1[m.queryIdx].pt for m in matches[:, 0]])
                .reshape(-1, 1, 2)
                .astype(np.float32)
            )
            dst = (
                np.asarray([kp2[m.trainIdx].pt for m in matches[:, 0]])
                .reshape(-1, 1, 2)
                .astype(np.float32)
            )
            homography_matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            self.homography_matrix = homography_matrix
            return homography_matrix
        else:
            logging.info("Not enough keypoints found!")
            self.homography_matrix = None
            return None

    def show_matches(self, im1, kp1, im2, kp2, good):
        im3 = cv2.drawMatchesKnn(
            im1,
            kp1,
            im2,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        plt.imshow(im3)
        plt.show()

    def update_coordinates_with_homography(self, coordinates, homography=None):
        """Updates the coordinates based on the homography

        Args:
            coordinates (np.ndarray): Initial coordinates
            homography (np.ndarray, Optional): Homography between two scenes.
            If it is not provided, it will use the latest calculated homography.

        Returns:
            new_coords (np.ndarray): Updated coordinates
        """
        if homography is None:
            homography = self.homography_matrix
            if homography is None:
                msg = (
                    "No homography could be found."
                    "Either calculate homography or provide a matrix."
                )
                logging.error(msg)
                raise ValueError(msg)

        # TODO: Enpand xy to have homogeneous coordinates
        mat = np.matmul(np.linalg.inv(homography), coordinates.T)
        mat = mat / mat[2]
        new_coords = mat.reshape(-1, mat.shape[1]).T

        return np.asarray(new_coords[:, :2])

    def phase_correlation(self, im1, im2):
        """Calculates the phase correlation between two images

        Args:
            im1 (np.ndarray[np.uint8]): Source Image, A 3D array of with shape HxWx3
            im2 (np.ndarray[np.uint8]): Destination Image,
            A 3D array of with shape HxWx3

        Returns:
            translation (np.ndarray): Array containing x-shift and y-shift between
            two images [x-shift, y-shift]
        """
        if len(im1.shape) > 2 and im1.shape[2] == 3:
            im1_1c = np.asarray(convert_BGR_to_GRAY(im1)).astype(np.float32)
            im2_1c = np.asarray(convert_BGR_to_GRAY(im2)).astype(np.float32)
        else:
            im1_1c, im2_1c = im1, im2

        ret, _ = cv2.phaseCorrelate(im1_1c, im2_1c)
        translation = np.asarray(
            [np.round(np.asarray(ret[0])), np.round(np.asarray(ret[1]))]
        )

        return translation

    def update_coordinates_with_translation(self, coordinates, translation):
        """Performs coordinate transform on a set of coordinates
        based on a translation vector

        Args:
            coordinates (np.ndarray): Initial coordinates in (x,y)
            translation (np.ndarray, Optional): Estimated translation between scene 1
            and scene 2.

        Returns:
            new_coords (np.ndarray): Updated coordinates
        """
        return coordinates - translation


class ResolutionEnhancer:
    """Class, one level above MultiViewGeometry, that handles the stitching together
    of sparse points for resolution enhancement. This also handles changes to
    the coordinate system to ignore unsampled regions and for downsampling that
    increases algorithm speed.

    User can select stitching by homography or phase correlation.
    """

    def __init__(
        self,
        sampling_coordinates_yx: np.ndarray,
        downsampling_factor: int = 1,
        mode="homography",
    ):
        """
        Args:
            sampling_coordinates_yx (np.ndarray): Sampling coordinates of sparse points
                in the coordinate system of the scene image.
            downsampling_factor (int, optional): Factor by which to resize
                the outputs by.  Defaults to 1.
            mode (str): Either 'homography' or 'phase_correlation'. Sets what backend
            to run resolution enhancement on.
        """
        if mode not in ["homography", "phase_correlation"]:
            raise ValueError(f"Selected mode {mode} is not valid.")
        else:
            self.mode = mode

        # Attributes for coordinate system modification
        self.downsampling_factor = downsampling_factor
        # Flag to save state upon first run
        self.first_frame_flag = True

        # Processor
        self.mvg = MultiViewGeometry()

        # Initialisation of coordinate system
        self.sampling_coords_yx, self.crop_limits = self._initialise_coordinate_system(
            sampling_coordinates_yx, 1 / self.downsampling_factor
        )
        self.bounds = self._get_bounds()
        self.sampling_coords_xy = self.sampling_coords_yx[:, ::-1]
        self.sampling_coords_homogeneous = self._cartesian_to_homogeneous(
            self.sampling_coords_xy
        )

        # State storage
        self.first_scene = None
        self.first_spectra = None
        self.points_xy = self.sampling_coords_xy.copy()
        self.values = None

        return

    def _initialise_coordinate_system(self, coords_yx, scale=1):
        """Defines the coordinate system used in resolution enhancement. This shifts
        and such that the min is 0 in both y and x. There is an option to
        multiplicatively scale the coordinate system after the shift.

        Args:
            coords_yx (np.ndarray): default sampling coordinates at default resolution
                of the scene frame.
            scale (int, optional): Scaling factor to downsample/upsample
            coordinates. Defaults to 1

        Returns:
            coords (np.ndarray): Updated coordinates in the (y, x) format.
            (Shape - [N, 2])
            offsets (np.ndarray): Crop-offsets in the form [y1, x1, y2, x2],
            where (x1, y1) is top-left and (x2, y2) is top-right
        """
        y1, y2, x1, x2 = (
            int(coords_yx[:, 0].min()),
            int(coords_yx[:, 0].max()),
            int(coords_yx[:, 1].min()),
            int(coords_yx[:, 1].max()),
        )
        coords_yx[:, 0] -= y1
        coords_yx[:, 1] -= x1
        coords_yx = (coords_yx * scale).astype(np.int32)
        offsets = np.array([y1, y2, x1, x2]).astype(np.int32)
        return coords_yx, offsets

    def _get_bounds(self):
        # Get recommended interpolation bounds
        y1, y2, x1, x2 = self.crop_limits
        h, w = int((y2 - y1) / self.downsampling_factor), int(
            (x2 - x1) / self.downsampling_factor
        )
        return np.array([0, w, 0, h])

    def _cartesian_to_homogeneous(self, arr):
        # Generate homogeneous coordinate system by adding row of ones
        return np.hstack((arr, np.ones((arr.shape[0], 1))))

    def _process_image(self, image):
        # Crops image according to values set in __init__
        cropped = image[
            self.crop_limits[0] : self.crop_limits[1],
            self.crop_limits[2] : self.crop_limits[3],
        ][:: self.downsampling_factor, :: self.downsampling_factor]

        # Add min max norm before processi
        result = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return result

    def _check_homography_successful(self, homography):
        """Inspects homography matrix to determine whether to incorporate this frame
        into the calculation or not.

        Args:
            homography (np.ndarray): Homography matrix shape (3,3)

        Returns:
            status (bool): True if result is good, False otherwise
        """
        # skipping the frame if the [h11, h12, h21, h22] differ by 1 in value or
        # no homography returned at all
        if homography is None or (
            (abs(np.round(homography[0][0]) - 1) >= 1)
            or (abs(np.round(homography[0][1]) - 0) >= 1)
            or (abs(np.round(homography[1][0]) - 0) >= 1)
            or (abs(np.round(homography[1][1]) - 1) >= 1)
        ):
            return False
        else:
            return True

    def process_next_frame(self, scene: np.ndarray, spectra: np.ndarray):
        """Processes the next scene frame, spectra pair from the Living Optics camera
        for resolution enhancement.

        Args:
            scene (np.ndarray): scene view image shape (H, W, 3)
            spectra (np.ndarray): spectra shape (N, N_channels)

        Returns:
            frame_success (bool): whether this frame has been incorporated or not
        """
        # Crop image if necessary
        scene = self._process_image(scene)

        if self.first_frame_flag:
            # Initialise array with information from first frame
            self.values = spectra.copy()
            self.first_scene, self.first_spectra = scene.copy(), spectra.copy()
            self.first_frame_flag = False
            frame_success = True

        elif self.mode == "homography":
            frame_success = self.process_next_frame_homography(scene, spectra)
        elif self.mode == "phase_correlation":
            frame_success = self.process_next_frame_phase_corr(scene, spectra)

        return frame_success

    def process_next_frame_homography(self, scene: np.ndarray, spectra: np.ndarray):
        """Processes the next scene frame, spectra pair from the Living Optics camera
        for resolution enhancement using the homography method.

        Args:
            scene (np.ndarray): scene view image shape (H, W, 3)
            spectra (np.ndarray): spectra shape (N, N_channels)

        Returns:
            frame_success (bool): whether this frame has been incorporated or not
        """
        homography_matrix = self.mvg.homography(self.first_scene, scene)
        frame_success = self._check_homography_successful(homography_matrix)
        if frame_success:
            new_coords = self.mvg.update_coordinates_with_homography(
                self.sampling_coords_homogeneous, homography_matrix
            )
            self.points_xy = np.append(self.points_xy, new_coords, axis=0)
            self.values = np.append(self.values, spectra, axis=0)

        return frame_success

    def process_next_frame_phase_corr(self, scene: np.ndarray, spectra: np.ndarray):
        """Processes the next scene frame, spectra pair from the Living Optics camera
        for resolution enhancement using the phase correlation method.

        Args:
            scene (np.ndarray): scene view image shape (H, W, 3)
            spectra (np.ndarray): spectra shape (N, N_channels)

        Returns:
            frame_success (bool): whether this frame has been incorporated or not
        """
        translation = self.mvg.phase_correlation(self.first_scene, scene)

        # No check for validity of translation for the time being
        frame_success = True

        if frame_success:
            new_coords = self.mvg.update_coordinates_with_translation(
                self.sampling_coords_xy, translation
            )
            self.points_xy = np.append(self.points_xy, new_coords, axis=0)
            self.values = np.append(self.values, spectra, axis=0)

        return frame_success

    def get_result(self, return_first_frame=False):
        # Gets result of resolution enhancement in form of a tuple of the coordinates
        # and values of the stitched-together sampled points.
        # User can then use this to interpolate in a downstream task.
        if return_first_frame:
            return (self.sampling_coords_xy, self.first_spectra)
        else:
            return (self.points_xy, self.values)


def get_rgb_wavelength_indices(wavelengths):
    """Returns RGB wavelength indices, corresponding to cone response
     from wavelengths in nm.

    "A wide range of colors can be obtained by mixing different amounts of red, green
    and blue light (additive color mixing). A possible combination of wavelengths is
    630 nm for red, 532 nm for green, and 465 nm for blue light."
    Source: https://www.rp-photonics.com/rgb_sources.html

    Args:
        wavelengths (np.ndarray): wavelengths array in nm shape (C).

    Returns:
        RGB Band (np.ndarray): List of RGB bands indices (in that order)
        closest to wavelengths specified.
    """
    return np.asarray([np.argmin(np.abs(wavelengths - i)) for i in [630, 532, 465]])


def convert_BGR_to_GRAY(frame):
    """ """
    return frame[:, :, 0] * 0.114 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.299


def run_resolution_enhancement(
    filepath, decode, frame_nos="all", downsampling_factor=3, mode="homography"
):
    """Sample function for resolution enhancement procedure that illustrates the usage
    of the API. Uses ResolutionEnhancer to stitch together consecutive frames and then
    does nearest neighbours interplolation of the spectra into a fixed region. Displays
    the 3-channel RGB visualisation of the cube.

    Args:
        filepath (str): Path to .loraw file
        decode (lo.sdk.api.acquisition.data.decode.SpectralDecoder): Living Optics
            SpectralDecoder object
        frame_nos (iterable of ints or 'all'): frames from file to run enhancement on.
            Defaults to 'all', which
            signifies to use all available frames.
        downsampling_factor (int): Factor which to downsample the native size of the
            scene frame image by.
        mode (str): Either 'homography' or 'phase_correlation'. Sets what backend to run
            resolution enhancement on.

    Returns:
        interp_highres (np.ndarray): Nearest Neighbours upsampled array for
            visualisation of the cube.
    """
    if type(downsampling_factor) is not int:
        raise TypeError("Downsampling factor must be of type integer.")

    print(f"Running resolution enhancement in {mode} mode.")

    enhance = ResolutionEnhancer(
        decode.sampling_coordinates, downsampling_factor=downsampling_factor, mode=mode
    )

    with sdkopen(filepath) as f:
        total_frames = len(f)
        if str(frame_nos) == "all":
            frame_nos = np.arange(0, total_frames)
        print(f"Total number of frames in the file: {total_frames}")
        for i in tqdm(frame_nos):
            f.seek(i)
            frame = f.read()

            info, scene, spectra = decode(frame, LORAWtoRGB8)
            frame_success = enhance.process_next_frame(scene, spectra)
            if not frame_success:
                print(f"Frame {i} failed, skipping.")

    # Spatial interpolation
    first_frame_result = enhance.get_result(return_first_frame=True)
    final_result = enhance.get_result()

    interp_first = Densifier(
        *enhance.bounds, first_frame_result[0], coordinates_are_yx=False
    )(first_frame_result[1])
    interp_highres = Densifier(
        *enhance.bounds, final_result[0], coordinates_are_yx=False
    )(final_result[1])


    
    return interp_highres, enhance.first_scene, interp_first
