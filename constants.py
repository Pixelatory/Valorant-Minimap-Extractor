from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# FLANN Algorithms.
FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_KDTREE_SINGLE = 4
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE_CUDA = 7

SUPPORTED_FLANN_ALGORITHMS = (
    FLANN_INDEX_LINEAR,
    FLANN_INDEX_KDTREE,
    FLANN_INDEX_KMEANS,
    FLANN_INDEX_COMPOSITE,
    FLANN_INDEX_KDTREE_SINGLE,
    FLANN_INDEX_HIERARCHICAL,
    FLANN_INDEX_LSH,
    FLANN_INDEX_KDTREE_CUDA,
)

# Standard maps.
MAP_HAVEN = "haven"
MAP_SPLIT = "split"
MAP_ASCENT = "ascent"
MAP_ICEBOX = "icebox"
MAP_BREEZE = "breeze"
MAP_FRACTURE = "fracture"
MAP_PEARL = "pearl"
MAP_LOTUS = "lotus"
MAP_SUNSET = "sunset"
MAP_ABYSS = "abyss"

# Team deathmatch maps.
MAP_DISTRICT = "district"
MAP_KASBAH = "kasbah"
MAP_PIAZZA = "piazza"
MAP_DRIFT = "drift"
MAP_GLITCH = "glitch"

# Practice map.
MAP_THE_RANGE = "the_range"

SUPPORTED_MAPS = (
    MAP_HAVEN,
    MAP_SPLIT,
    MAP_ASCENT,
    MAP_ICEBOX,
    MAP_BREEZE,
    MAP_FRACTURE,
    MAP_PEARL,
    MAP_LOTUS,
    MAP_SUNSET,
    MAP_ABYSS,
    MAP_DISTRICT,
    MAP_KASBAH,
    MAP_PIAZZA,
    MAP_DRIFT,
    MAP_GLITCH,
    MAP_THE_RANGE,
)

# From OpenCV's imread().
# https://docs.opencv.org/4.10.0/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8.
SUPPORTED_FILE_EXTS = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".avif",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".pfm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
    ".pic",
)
