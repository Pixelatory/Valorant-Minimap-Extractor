#!/usr/bin/python3
import argparse
import os

import cv2

import constants
import feature_match


def gather_file_paths(directory: str) -> list[str]:
    """Gather all file paths in a given directory."""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def gather_supported_files(filenames: list[str]) -> list[str]:
    """Gather files that have a supported file extension."""
    valid_files = []
    for filename in filenames:
        for file_ext in constants.SUPPORTED_FILE_EXTS:
            if filename.endswith(file_ext):
                valid_files.append(filename)
    return valid_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ValorantMinimapExtractor",
        description="Extracts the minimap from a screenshot of a Valorant game.",
        add_help=True,
    )
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "filenames", nargs="*", type=str, help="One or more filenames to process."
    )
    general_group.add_argument(
        "--map",
        help="The Valorant map(s) played in the given filenames.",
        choices=constants.SUPPORTED_MAPS,
        type=str,
        required=True,
    )
    general_group.add_argument(
        "--mode",
        help="Mode to extract Valorant minimap. Either 'boundary' or 'exact'.",
        choices=("boundary", "exact"),
        type=str,
        required=True,
    )
    general_group.add_argument(
        "--input-dir", type=str, help="Directory of files to process."
    )
    general_group.add_argument(
        "--output-dir", type=str, help="Directory to save output files in."
    )
    general_group.add_argument("--debug", action="store_true")

    sift_group = parser.add_argument_group("SIFT Options")
    sift_group.add_argument(
        "--num-features",
        type=int,
        help="The number of best features to retain. 0 means all features are retained.",
        default=0,
    )
    sift_group.add_argument(
        "--num-octave-layers",
        type=int,
        default=3,
    )
    sift_group.add_argument(
        "--contrast-threshold",
        type=float,
        help="The contrast threshold used to filter out weak features in semi-uniform "
        "(low-contrast) regions.",
        default=0.04,
    )
    sift_group.add_argument(
        "--edge-threshold",
        type=float,
        help="The threshold used to filter out edge-like features.",
        default=10.0,
    )
    sift_group.add_argument(
        "--sigma",
        type=float,
        help="The sigma of the Gaussian applied to the input image at the octave 0.",
        default=1.6,
    )

    flann_group = parser.add_argument_group("FLANN Matcher Options")
    flann_group.add_argument(
        "--algorithm",
        type=int,
        help="The FLANN algorithm to use.",
        choices=constants.SUPPORTED_FLANN_ALGORITHMS,
        default=constants.FLANN_INDEX_KDTREE,
    )
    flann_group.add_argument(
        "--checks",
        type=int,
        help="Number of times the tree(s) in the index should be recursively traversed.",
        default=50,
    )
    flann_group.add_argument(
        "--k",
        type=int,
        help="How many closest matches should be returned for each query descriptor.",
        default=2,
    )
    args = parser.parse_args()

    mode = args.mode.lower()

    # Validate arguments.
    if not args.filenames and not args.input_dir:
        raise ValueError("One of 'filenames' or 'input_dir' should be provided.")

    # Gather filenames with supported extensions.
    if args.filenames:
        filenames = gather_supported_files(args.filenames)
    else:
        filenames = gather_supported_files(gather_file_paths(args.input_dir))

    if not filenames:
        raise ValueError(
            f"No supported files provided. "
            "Supported file extensions: {SUPPORTED_FILE_EXTS}."
        )

    for filename in filenames:
        sift_module = feature_match.SIFTModule(
            num_features=args.num_features,
            num_octave_layers=args.num_octave_layers,
            contrast_threshold=args.contrast_threshold,
            edge_threshold=args.edge_threshold,
            sigma=args.sigma,
        )
        flann_matcher = feature_match.FLANNMatcherModule(
            algorithm=args.algorithm,
            checks=args.checks,
            k=args.k,
        )
        extract_fn = feature_match.MODE_TO_FN[args.mode]
        extracted_minimap = extract_fn(
            full_img_path=filename,
            valorant_map=args.map,
            keypoint_module=sift_module,
            matcher_module=flann_matcher,
            debug=args.debug,
        )
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base_name = os.path.basename(filename)
            name, ext = os.path.splitext(base_name)
            save_path = os.path.join(args.output_dir, name + "-minimap" + ext)
        else:
            base_name = os.path.basename(filename)
            name, ext = os.path.splitext(base_name)
            base_dir = os.path.dirname(filename)
            save_path = os.path.join(base_dir, name + "-minimap" + ext)
        cv2.imwrite(save_path, extracted_minimap)
