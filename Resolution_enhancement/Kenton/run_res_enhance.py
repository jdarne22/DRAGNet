import os
from pathlib import Path

import numpy as np
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from res_enhance import run_resolution_enhancement

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Living Optics Resolution Enhancement CLI",
        description="",
        epilog="Living Optics 2024",
    )

    parser.add_argument("--file", type=str, help="Path to loraw datafile.")
    parser.add_argument("--calibration", type=str, help="Path to calibration folder.")
    parser.add_argument(
        "--filter_snap", type=str, help="Path to 600 nm filter snap file."
    )
    parser.add_argument(
        "--downsampling_factor",
        nargs="?",
        default=3,
        type=int,
        help="Integer downsampling factor.",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        default="homography",
        type=str,
        help="Backend to run resolution enhancement in. Either 'homography' "
        "or 'phase_correlation'. Defaults to 'homography'",
    )
    args = parser.parse_args()

    # Create decoder object, the filter_snap is optional for field calibration
    decode = SpectralDecoder.from_calibration(args.calibration, args.filter_snap)
    print(args.downsampling_factor)

    im = run_resolution_enhancement(
        args.file, decode, "all", args.downsampling_factor, args.mode
    )
    # Saveout to numpy array
    np.save(
        os.path.join(
            Path(args.file).parent.absolute(), Path(args.file).stem + "-enhanced.npy"
        ),
        im,
    )
