import os
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike

import constants


class KeypointsModule:
    def __call__(
        self, img_gray: MatLike, template_gray: MatLike
    ) -> tuple[Any, Any, Any, Any]:
        raise NotImplementedError()


class SIFTModule(KeypointsModule):
    def __init__(
        self,
        num_features: int = 0,
        num_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6,
    ):
        """
        The SIFT keypoints extractor.

        This documentation is largely copied from
        https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html. Default values
        are the same defaults from OpenCV.

        Note: The contrast threshold will be divided by num_octave_layers when
        filtering is applied. When num_octave_layers is set to default and if you want
        to use the value used in D. Lowe paper, 0.03, set this argument to 0.09.

        :param num_features: The number of best features to retain. The features
            are ranked by their scores (measured in SIFT algorithm as the local
            contrast). 0 means all features are retained.
        :param num_octave_layers: The number of layers in each octave. 3 is the
            value used in the D. Lowe paper. The number of octaves is computed
            automatically from the image resolution.
        :param contrast_threshold: The contrast threshold used to filter out weak
            features in semi-uniform (low-contrast) regions. The larger the threshold,
            the less features are produced by the detector.
        :param edge_threshold: The threshold used to filter out edge-like features.
            Note that the its meaning is different from the contrastThreshold, i.e.
            the larger the edgeThreshold, the less features are filtered out (more
            features are retained).
        :param sigma: The sigma of the Gaussian applied to the input image at the
            octave 0. If your image is captured with a weak camera with soft lenses,
            you might want to reduce the number.
        """
        self.sift = cv2.SIFT_create(
            nfeatures=num_features,
            nOctaveLayers=num_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    def __call__(self, img_gray, template_gray):
        """
        :returns: Tuple of full image keypoints, full image descriptors,
            template image keypoints, template image descriptors.
        """
        keypoints_template, descriptors_template = self.sift.detectAndCompute(
            template_gray, None
        )
        keypoints_full, descriptors_full = self.sift.detectAndCompute(img_gray, None)

        return (
            keypoints_full,
            descriptors_full,
            keypoints_template,
            descriptors_template,
        )


class MatcherModule:
    def __call__(self, descriptors_img: MatLike, descriptors_template: MatLike):
        raise NotImplementedError()


class FLANNMatcherModule(MatcherModule):
    def __init__(
        self,
        algorithm: int = constants.FLANN_INDEX_KDTREE,
        checks: int = 50,
        k: int = 2,
        index_params: dict[str, Any] | None = None,
    ):
        """
        The FLANN feature matcher.

        :param algorithm: Integer corresponding to the FLANN algorithm to use.
            View `constants.py` for algorithm options.
        :param checks: Number of times the tree(s) in the index should be
            recursively traversed. Higher values improve accuracy at the
            cost of increased computation time.
        :param k: Determines how many closest matches should be returned
            for each query descriptor.
        """
        if index_params is None:
            index_params = {}
        index_params = dict(algorithm=algorithm, **index_params)
        search_params = dict(checks=checks)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.k = k

    def __call__(
        self,
        descriptors_img,
        descriptors_template,
    ):
        # Find matches
        return self.flann.knnMatch(descriptors_template, descriptors_img, k=self.k)


def match_filter(
    matches,
    matches_filter_threshold: float = 0.75,
):
    """
    Lowe's ratio test to filter for good matches, extended for
    k values greater than 2 in KNN algorithm.

    :param matches: The matches to filter.
    :param matches_filter_threshold: A value between 0-1. Filter out
        matched points if the ratio between their distances is too
        far apart. For fewer, but more confident matches, use a lower
        threshold. Vice versa for higher threshold.
    """
    good_matches = []
    for match_list in matches:
        # Check if the closest match is significantly closer than the others.
        m = match_list[0]
        min_distance = m.distance
        second_min_distance = float("inf")

        for i in range(1, len(match_list)):
            n = match_list[i]
            if n.distance < second_min_distance:
                second_min_distance = n.distance

        # Apply the ratio test between the closest and second closest neighbors.
        if min_distance < matches_filter_threshold * second_min_distance:
            good_matches.append(m)
    return good_matches


def extract_minimap_exact(
    full_img_path: str,
    valorant_map: str,
    keypoint_module: KeypointsModule,
    matcher_module: MatcherModule,
    matches_filter_threshold: float = 0.75,
    debug: bool = False,
):
    """
    :param minimap_img_path: The minimap outline image to use, for
        extracting the minimap. Should be a PNG file with some
        transparency.
    :param valorant_map: String corresponding to the valorant map
        played in `full_img_path`.
    :param matches_filter_threshold: A value between 0-1. Filter out
        matched points if the ratio between their distances is too
        far apart. For fewer, but more confident matches, use a lower
        threshold. Vice versa for higher threshold.
    """
    valorant_map = valorant_map.lower()
    if valorant_map not in constants.SUPPORTED_MAPS:
        raise ValueError(f"{valorant_map} is not a supported map.")

    # Load full-size image.
    full_img_rgb = cv2.imread(full_img_path)
    full_img_gray = cv2.cvtColor(full_img_rgb, cv2.COLOR_BGR2GRAY)

    # Load template with alpha channel.
    template = cv2.imread(
        os.path.join(constants.BASE_DIR, f"map_templates/{valorant_map}_minimap.png"),
        cv2.IMREAD_UNCHANGED,
    )

    # Separate template channels.
    if template.shape[2] == 4:
        template_rgb = template[:, :, :3]
        template_alpha = template[:, :, 3]
    else:
        raise ValueError("Minimap image file does not have an alpha channel!")

    # Convert template to grayscale for feature detection.
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors.
    (
        keypoints_full,
        descriptors_full,
        keypoints_template,
        descriptors_template,
    ) = keypoint_module(full_img_gray, template_gray)

    # Find matches.
    matches = matcher_module(descriptors_full, descriptors_template)

    # Apply Lowe's ratio test to filter for good matches.
    good_matches = match_filter(matches, matches_filter_threshold)

    if not good_matches:
        raise ValueError("No good matches found.")

    if debug:
        os.makedirs(os.path.join(constants.BASE_DIR, "debug/"), exist_ok=True)
        result = cv2.drawMatches(
            template_gray,
            keypoints_template,
            full_img_gray,
            keypoints_full,
            good_matches[:20],
            None,
            flags=2,
        )
        cv2.imwrite(os.path.join(constants.BASE_DIR, "debug/matches.png"), result)

    # Extract matching keypoints.
    pts_template = np.float32(
        [keypoints_template[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts_full = np.float32(
        [keypoints_full[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Estimate Affine transformation.
    affine_matrix, _ = cv2.estimateAffine2D(pts_template, pts_full)

    # Apply the transformation to the template and its alpha channel.
    rows, cols = full_img_gray.shape
    transformed_alpha = cv2.warpAffine(template_alpha, affine_matrix, (cols, rows))
    transformed_template = cv2.warpAffine(template_gray, affine_matrix, (cols, rows))

    if debug:
        os.makedirs(os.path.join(constants.BASE_DIR, "debug/"), exist_ok=True)
        cv2.imwrite(
            os.path.join(constants.BASE_DIR, "debug/transformed_template.png"),
            transformed_template,
        )

        # Construct the overlay image and save it to debug directory.
        _, mask = cv2.threshold(transformed_template, 1, 255, cv2.THRESH_BINARY)
        template_color = cv2.cvtColor(transformed_template, cv2.COLOR_GRAY2BGR)
        template_masked = cv2.bitwise_and(template_color, template_color, mask=mask)
        full_image_with_overlay = cv2.addWeighted(
            full_img_rgb, 1, template_masked, 0.5, 0
        )
        cv2.imwrite(
            os.path.join(constants.BASE_DIR, "debug/overlay.png"),
            full_image_with_overlay,
        )

    # Create a binary mask from the transformed alpha channel.
    _, mask = cv2.threshold(transformed_alpha, 1, 255, cv2.THRESH_BINARY)

    # Extract the corresponding region from the full image.
    full_img_rgba = cv2.cvtColor(full_img_rgb, cv2.COLOR_RGB2RGBA)
    rect = cv2.boundingRect(transformed_alpha)
    x, y, w, h = rect

    extracted_minimap_img = cv2.bitwise_and(full_img_rgba, full_img_rgba, mask=mask)
    return extracted_minimap_img[y : y + h, x : x + w]


def extract_minimap_from_boundary(
    full_img_path: str,
    valorant_map: str,
    keypoint_module: KeypointsModule,
    matcher_module: MatcherModule,
    matches_filter_threshold: float = 0.75,
    padding: int = 0,
    debug: bool = False,
):
    """
    :param minimap_img_path: The minimap outline image to use, for
        extracting the minimap. Should be a PNG file with some
        transparency.
    :param valorant_map: String corresponding to the valorant map
        played in `full_img_path`.
    :param matches_filter_threshold: A value between 0-1. Filter out
        matched points if the ratio between their distances is too
        far apart. For fewer, but more confident matches, use a lower
        threshold. Vice versa for higher threshold.
    """
    valorant_map = valorant_map.lower()
    if valorant_map not in constants.SUPPORTED_MAPS:
        raise ValueError(f"{valorant_map} is not a supported map.")

    # Load full-size and template images.
    full_img_rgb = cv2.imread(full_img_path)
    full_img_gray = cv2.cvtColor(full_img_rgb, cv2.COLOR_BGR2GRAY)

    # Load template with alpha channel.
    template = cv2.imread(
        os.path.join(constants.BASE_DIR, f"map_templates/{valorant_map}_minimap.png"),
        cv2.IMREAD_UNCHANGED,
    )

    # Separate template channels.
    if template.shape[2] == 4:
        template_bgr = template[:, :, :3]
        template_alpha = template[:, :, 3]
    else:
        raise ValueError("Minimap image file does not have an alpha channel!")

    # Convert template to grayscale for feature detection.
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors.
    (
        keypoints_full,
        descriptors_full,
        keypoints_template,
        descriptors_template,
    ) = keypoint_module(full_img_gray, template_gray)

    if len(keypoints_template) == 0:
        raise ValueError("No keypoints found in the template image.")

    # Find matches.
    matches = matcher_module(descriptors_full, descriptors_template)

    # Apply Lowe's ratio test to filter for good matches.
    good_matches = match_filter(matches, matches_filter_threshold)

    if not good_matches:
        raise ValueError("No good matches found.")

    if debug:
        os.makedirs(os.path.join(constants.BASE_DIR, "debug/"), exist_ok=True)
        result = cv2.drawMatches(
            template_gray,
            keypoints_template,
            full_img_gray,
            keypoints_full,
            good_matches[:20],
            None,
            flags=2,
        )
        cv2.imwrite(os.path.join(constants.BASE_DIR, "debug/matches.png"), result)

    invalid_indices = [
        m.queryIdx for m in good_matches if m.queryIdx >= len(keypoints_template)
    ]
    if invalid_indices:
        raise ValueError(f"Invalid query indices in good_matches: {invalid_indices}")

    pts_template = np.float32(
        [keypoints_template[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts_full = np.float32(
        [keypoints_full[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Estimate Affine transformation.
    affine_matrix, _ = cv2.estimateAffine2D(pts_template, pts_full)

    rows, cols = full_img_gray.shape
    template_alpha = cv2.warpAffine(template_alpha, affine_matrix, (cols, rows))
    template_gray = cv2.warpAffine(template_gray, affine_matrix, (cols, rows))
    rect = cv2.boundingRect(template_alpha)
    x, y, w, h = rect

    if debug:
        os.makedirs(os.path.join(constants.BASE_DIR, "debug/"), exist_ok=True)
        cv2.imwrite(
            os.path.join(constants.BASE_DIR, "debug/transformed_template.png"),
            template_gray,
        )

        # Construct the overlay image and save it to debug directory.
        _, mask = cv2.threshold(template_gray, 1, 255, cv2.THRESH_BINARY)
        template_color = cv2.cvtColor(template_gray, cv2.COLOR_GRAY2BGR)
        template_masked = cv2.bitwise_and(template_color, template_color, mask=mask)
        full_image_with_overlay = cv2.addWeighted(
            full_img_rgb, 1, template_masked, 0.5, 0
        )
        cv2.imwrite(
            os.path.join(constants.BASE_DIR, "debug/overlay.png"),
            full_image_with_overlay,
        )

    return full_img_rgb[y - padding : y + h + padding, x - padding : x + w + padding]


MODE_TO_FN = {
    "boundary": extract_minimap_from_boundary,
    "exact": extract_minimap_exact,
}
