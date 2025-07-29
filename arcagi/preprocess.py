import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def expand_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Expands a given 2D list (matrix) to a fixed 30x30 matrix.
    The original matrix is centered within the 30x30 grid.
    Positions beyond the size of the original matrix are filled with -1.
    """
    # Create empty 30x30 matrix filled with -1
    new_matrix: List[List[int]] = [[-1 for _ in range(30)] for _ in range(30)]

    # Calculate dimensions of input matrix
    height: int = len(matrix)
    width: int = len(matrix[0]) if height > 0 else 0

    # Calculate starting positions to center the matrix
    start_row: int = (30 - height) // 2
    start_col: int = (30 - width) // 2

    # Copy input matrix to centered position in the new matrix
    for i, row in enumerate(matrix):
        if i >= height:
            break
        for j, val in enumerate(row):
            if j >= width:
                break
            new_matrix[start_row + i][start_col + j] = val

    return new_matrix


def apply_d4_transformations(matrix: List[List[int]]) -> List[List[List[int]]]:
    """
    Apply all 8 D4 symmetry transformations to a matrix.

    The D4 group consists of:
    - 4 rotations: 0°, 90°, 180°, 270°
    - 4 reflections: horizontal flip + above rotations

    Args:
        matrix: 2D list representing the input matrix

    Returns:
        List of 8 transformed matrices in the following order:
        [identity, rot90, rot180, rot270, flip_h, flip_h_rot90, flip_h_rot180, flip_h_rot270]
    """

    # Helper function to rotate matrix 90 degrees clockwise
    def rotate_90(mat: List[List[int]]) -> List[List[int]]:
        h = len(mat)
        w = len(mat[0]) if h > 0 else 0
        rotated = [[0 for _ in range(h)] for _ in range(w)]
        for i in range(h):
            for j in range(w):
                rotated[j][h - 1 - i] = mat[i][j]
        return rotated

    # Helper function to flip matrix horizontally
    def flip_horizontal(mat: List[List[int]]) -> List[List[int]]:
        return [row[::-1] for row in mat]

    # Generate all 8 transformations
    transformations: List[List[List[int]]] = []

    # 1. Identity (no transformation)
    transformations.append([row[:] for row in matrix])

    # 2. Rotate 90° clockwise
    rot90 = rotate_90(matrix)
    transformations.append(rot90)

    # 3. Rotate 180°
    rot180 = rotate_90(rot90)
    transformations.append(rot180)

    # 4. Rotate 270° clockwise (or 90° counter-clockwise)
    rot270 = rotate_90(rot180)
    transformations.append(rot270)

    # 5. Horizontal flip
    flip_h = flip_horizontal(matrix)
    transformations.append(flip_h)

    # 6. Horizontal flip + rotate 90°
    flip_h_rot90 = rotate_90(flip_h)
    transformations.append(flip_h_rot90)

    # 7. Horizontal flip + rotate 180°
    flip_h_rot180 = rotate_90(flip_h_rot90)
    transformations.append(flip_h_rot180)

    # 8. Horizontal flip + rotate 270°
    flip_h_rot270 = rotate_90(flip_h_rot180)
    transformations.append(flip_h_rot270)

    return transformations


def compute_order2_features(matrix: NDArray[np.int32]) -> NDArray[np.uint8]:
    """
    Computes order-2 (pairwise) relational features for a 30x30 matrix.

    Returns features of shape (30, 30, 44) where:
    - First 36 features: pairwise color matching in 3x3 neighborhood (0=same, 1=different)
    - Last 8 features: center-to-neighbor mask detection (0=neighbor is mask, 1=neighbor is not mask)

    Args:
        matrix: numpy array of shape (30, 30) with color values

    Returns:
        features: numpy array of shape (30, 30, 44) with binary features
    """
    height, width = matrix.shape
    features = np.zeros((height, width, 44), dtype=np.uint8)

    # Define 3x3 neighborhood offsets (row, col)
    neighborhood_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # 8 neighbor offsets (excluding center)
    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for i in range(height):
        for j in range(width):
            # Get 3x3 neighborhood colors
            neighborhood_colors: List[int] = []
            for di, dj in neighborhood_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    neighborhood_colors.append(int(matrix[ni, nj]))
                else:
                    # Out of bounds treated as mask (-1)
                    neighborhood_colors.append(-1)

            # Feature 1: Pairwise color matching (36 features)
            feature_idx = 0
            for idx1 in range(9):
                for idx2 in range(idx1 + 1, 9):
                    color1: int = neighborhood_colors[idx1]
                    color2: int = neighborhood_colors[idx2]
                    # 0 if same color, 1 if different color
                    features[i, j, feature_idx] = 1 if color1 != color2 else 0
                    feature_idx += 1

            # Feature 2: Center-to-neighbor mask detection (8 features)
            for idx, (di, dj) in enumerate(neighbor_offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_color = int(matrix[ni, nj])
                else:
                    neighbor_color = -1  # Out of bounds treated as mask

                # 0 if neighbor is mask (-1), 1 if neighbor is not mask
                features[i, j, 36 + idx] = 0 if neighbor_color == -1 else 1

    return features


def compute_order3_features(matrix: NDArray[np.int32]) -> NDArray[np.uint8]:
    """
    Computes order-3 (triplet) relational features for a 30x30 matrix.

    Returns features of shape (30, 30, 84) where each feature represents
    whether a triplet of cells in the 3x3 neighborhood all have the same color.
    There are C(9,3) = 84 possible triplets in a 3x3 neighborhood.

    Args:
        matrix: numpy array of shape (30, 30) with color values

    Returns:
        features: numpy array of shape (30, 30, 84) with binary features
    """
    height, width = matrix.shape
    features = np.zeros((height, width, 84), dtype=np.uint8)

    # Define 3x3 neighborhood offsets (row, col)
    neighborhood_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # Generate all combinations of 3 positions from the 9 neighborhood positions
    # This gives us C(9,3) = 84 triplets
    from itertools import combinations

    triplet_indices = list(combinations(range(9), 3))

    # For each cell in the matrix
    for i in range(height):
        for j in range(width):
            # Get colors of all 9 neighbors (including center)
            neighbor_colors: List[int] = []
            for di, dj in neighborhood_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_colors.append(int(matrix[ni, nj]))
                else:
                    neighbor_colors.append(-1)  # Out of bounds = mask

            # Compute features for all 84 triplets
            for feat_idx, (idx1, idx2, idx3) in enumerate(triplet_indices):
                color1: int = neighbor_colors[idx1]
                color2: int = neighbor_colors[idx2]
                color3: int = neighbor_colors[idx3]

                # Feature is 1 if all three colors are the same (and not mask)
                if color1 == color2 == color3 and color1 != -1:
                    features[i, j, feat_idx] = 1
                else:
                    features[i, j, feat_idx] = 0

    return features


def compute_ncolors_features(matrix: NDArray[np.int32]) -> NDArray[np.uint8]:
    """
    Computes boolean features indicating the number of distinct colors in the 3x3 neighborhood.

    Returns features of shape (30, 30, 9) where each of the 9 features indicates
    whether there are at least that many distinct colors in the 3x3 neighborhood.
    Colors with value -1 (mask) are excluded from the count.

    For example:
    - 1 distinct color: [1, 0, 0, 0, 0, 0, 0, 0, 0]
    - 2 distinct colors: [1, 1, 0, 0, 0, 0, 0, 0, 0]
    - 3 distinct colors: [1, 1, 1, 0, 0, 0, 0, 0, 0]
    - 9 distinct colors: [1, 1, 1, 1, 1, 1, 1, 1, 1]

    Args:
        matrix: numpy array of shape (30, 30) with color values

    Returns:
        features: numpy array of shape (30, 30, 9) with boolean color count features
    """
    height, width = matrix.shape
    features = np.zeros((height, width, 9), dtype=np.uint8)

    # Define 3x3 neighborhood offsets (row, col)
    neighborhood_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # For each cell in the matrix
    for i in range(height):
        for j in range(width):
            # Get colors of all 9 neighbors (including center)
            neighbor_colors: Set[int] = set()
            for di, dj in neighborhood_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    color = int(matrix[ni, nj])
                    if color != -1:  # Exclude mask values
                        neighbor_colors.add(color)
                # Note: out-of-bounds cells are ignored (not counted as mask)

            # Count distinct colors (excluding mask)
            ncolors = len(neighbor_colors)

            # Set boolean features: 1 for positions up to ncolors, 0 for rest
            for k in range(9):
                if k + 1 <= ncolors:  # k+1 because positions are 1-indexed
                    features[i, j, k] = 1
                else:
                    features[i, j, k] = 0

    return features


def compute_ncells_matching_center_features(
    matrix: NDArray[np.int32],
) -> NDArray[np.uint8]:
    """
    Computes boolean features indicating the number of cells in the 3x3 neighborhood that match the center cell's color.

    Returns features of shape (30, 30, 9) where each of the 9 features indicates
    whether there are at least that many cells matching the center cell's color.
    The count includes the center cell itself and ranges from 1 to 9.

    For example:
    - 1 matching cell: [1, 0, 0, 0, 0, 0, 0, 0, 0]
    - 2 matching cells: [1, 1, 0, 0, 0, 0, 0, 0, 0]
    - 3 matching cells: [1, 1, 1, 0, 0, 0, 0, 0, 0]
    - 9 matching cells: [1, 1, 1, 1, 1, 1, 1, 1, 1]

    Args:
        matrix: numpy array of shape (30, 30) with color values

    Returns:
        features: numpy array of shape (30, 30, 9) with boolean matching cell count features
    """
    height, width = matrix.shape
    features = np.zeros((height, width, 9), dtype=np.uint8)

    # Define 3x3 neighborhood offsets (row, col)
    neighborhood_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # For each cell in the matrix
    for i in range(height):
        for j in range(width):
            center_color = int(matrix[i, j])
            matching_count = 0

            # Count cells in neighborhood that match center color
            for di, dj in neighborhood_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    if int(matrix[ni, nj]) == center_color:
                        matching_count += 1
                # Note: out-of-bounds cells are ignored (not counted)

            # Set boolean features: 1 for positions up to matching_count, 0 for rest
            for k in range(9):
                if k + 1 <= matching_count:  # k+1 because positions are 1-indexed
                    features[i, j, k] = 1
                else:
                    features[i, j, k] = 0

    return features


def compute_is_mask_features(matrix: NDArray[np.int32]) -> NDArray[np.uint8]:
    """
    Computes a binary mask feature for each cell.

    Returns features of shape (30, 30, 1) where the single feature is:
    - 1 if the cell is a mask (value -1 or 10)
    - 0 if the cell is a regular color (values 0-9, excluding 10)

    Args:
        matrix: numpy array of shape (30, 30) with color values

    Returns:
        features: numpy array of shape (30, 30, 1) with binary mask features
    """
    height, width = matrix.shape
    features = np.zeros((height, width, 1), dtype=np.uint8)

    # Set feature to 1 for mask values (-1 or 10)
    mask_condition = (matrix == -1) | (matrix == 10)
    features[:, :, 0] = mask_condition.astype(np.uint8)

    return features


def load_data_from_directory(
    directory: str,
    filename_filter: Optional[str] = None,
    feature_order2: bool = False,
    feature_order3: bool = False,
    feature_ncolors: bool = False,
    feature_is_mask: bool = False,
    feature_ncells_matching_center: bool = False,
    augment_d4_symmetry: bool = False,
) -> Tuple[
    List[str],
    List[int],
    List[bool],
    NDArray[np.int32],
    NDArray[np.int32],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.uint8]],
]:
    """
    Loads JSON files from a given directory.
    For each JSON file, it reads BOTH "train" and "test" subsets,
    expands the input and output matrices, and aggregates the data.

    Args:
        directory: Directory containing JSON files
        filename_filter: Optional filter for specific filenames
        feature_order2: Whether to compute order-2 (pairwise) relational features
        feature_order3: Whether to compute order-3 (triplet) relational features
        feature_ncolors: Whether to compute number of distinct colors in 3x3 neighborhood
        feature_is_mask: Whether to compute binary mask feature (1 if cell is mask, 0 otherwise)
        feature_ncells_matching_center: Whether to compute number of cells matching center cell color
        augment_d4_symmetry: Whether to apply all 8 D4 symmetry transformations to augment data (8x increase)

    Returns:
        filenames: List of filename strings
        indices: List of example indices within each file
        subset_example_index_is_train: List of booleans indicating if example is from "train" subset (True) or "test" subset (False)
        inputs_array: numpy array of shape (n_examples, 30, 30) with int32 dtype
        outputs_array: numpy array of shape (n_examples, 30, 30) with int32 dtype
        inputs_order2: numpy array of shape (n_examples, 30, 30, 44) with uint8 dtype (if feature_order2=True, else None)
        outputs_order2: numpy array of shape (n_examples, 30, 30, 44) with uint8 dtype (if feature_order2=True, else None)
        inputs_order3: numpy array of shape (n_examples, 30, 30, 84) with uint8 dtype (if feature_order3=True, else None)
        outputs_order3: numpy array of shape (n_examples, 30, 30, 84) with uint8 dtype (if feature_order3=True, else None)
        inputs_ncolors: numpy array of shape (n_examples, 30, 30, 9) with uint8 dtype (if feature_ncolors=True, else None)
        outputs_ncolors: numpy array of shape (n_examples, 30, 30, 9) with uint8 dtype (if feature_ncolors=True, else None)
        inputs_is_mask: numpy array of shape (n_examples, 30, 30, 1) with uint8 dtype (if feature_is_mask=True, else None)
        outputs_is_mask: numpy array of shape (n_examples, 30, 30, 1) with uint8 dtype (if feature_is_mask=True, else None)
        inputs_ncells_matching_center: numpy array of shape (n_examples, 30, 30, 9) with uint8 dtype (if feature_ncells_matching_center=True, else None)
        outputs_ncells_matching_center: numpy array of shape (n_examples, 30, 30, 9) with uint8 dtype (if feature_ncells_matching_center=True, else None)
    """
    filenames: List[str] = []
    indices: List[int] = []
    subset_example_index_is_train: List[bool] = []
    inputs_expanded: List[List[List[int]]] = []  # Each inner element is 30x30.
    outputs_expanded: List[List[List[int]]] = []

    path: Path = Path(directory)
    json_files: List[Path] = sorted(list(path.glob("*.json")))

    # Always load both train and test subsets
    for json_file in tqdm(json_files, desc=f"Loading data from {directory}"):
        with json_file.open("r") as f:
            data: Dict[str, Any] = json.load(f)

        # Process "train" subset if it exists
        if "train" in data:
            examples: List[Dict[str, Any]] = data["train"]
            for idx, example in enumerate(examples):
                input_colors: List[List[int]] = example["input"]
                output_colors: List[List[int]] = example["output"]

                if augment_d4_symmetry:
                    # Apply all 8 D4 transformations
                    input_transforms = apply_d4_transformations(input_colors)
                    output_transforms = apply_d4_transformations(output_colors)

                    for transform_idx in range(8):
                        filenames.append(json_file.name)
                        indices.append(idx)
                        subset_example_index_is_train.append(True)
                        inputs_expanded.append(
                            expand_matrix(input_transforms[transform_idx])
                        )
                        outputs_expanded.append(
                            expand_matrix(output_transforms[transform_idx])
                        )
                else:
                    # No augmentation, just add the original
                    filenames.append(json_file.name)
                    indices.append(idx)
                    subset_example_index_is_train.append(True)
                    inputs_expanded.append(expand_matrix(input_colors))
                    outputs_expanded.append(expand_matrix(output_colors))

        # Process "test" subset if it exists
        if "test" in data:
            examples: List[Dict[str, Any]] = data["test"]
            # Continue numbering from where train left off
            train_count = len(data.get("train", []))
            for idx, example in enumerate(examples):
                input_colors: List[List[int]] = example["input"]
                output_colors: List[List[int]] = example["output"]

                if augment_d4_symmetry:
                    # Apply all 8 D4 transformations
                    input_transforms = apply_d4_transformations(input_colors)
                    output_transforms = apply_d4_transformations(output_colors)

                    for transform_idx in range(8):
                        filenames.append(json_file.name)
                        indices.append(train_count + idx)
                        subset_example_index_is_train.append(False)
                        inputs_expanded.append(
                            expand_matrix(input_transforms[transform_idx])
                        )
                        outputs_expanded.append(
                            expand_matrix(output_transforms[transform_idx])
                        )
                else:
                    # No augmentation, just add the original
                    filenames.append(json_file.name)
                    indices.append(train_count + idx)
                    subset_example_index_is_train.append(False)
                    inputs_expanded.append(expand_matrix(input_colors))
                    outputs_expanded.append(expand_matrix(output_colors))

    # Filter by filename if specified
    if filename_filter is not None:
        # Create mask for matching filenames
        mask: List[bool] = [filename == filename_filter for filename in filenames]

        # Apply mask to filter data
        filenames = [filenames[i] for i in range(len(filenames)) if mask[i]]
        indices = [indices[i] for i in range(len(indices)) if mask[i]]
        subset_example_index_is_train = [
            subset_example_index_is_train[i]
            for i in range(len(subset_example_index_is_train))
            if mask[i]
        ]

        # Filter expanded inputs and outputs before converting to arrays
        inputs_expanded = [
            inputs_expanded[i] for i in range(len(inputs_expanded)) if mask[i]
        ]
        outputs_expanded = [
            outputs_expanded[i] for i in range(len(outputs_expanded)) if mask[i]
        ]

    # Convert to numpy arrays with int32 dtype
    inputs_array = np.array(inputs_expanded, dtype=np.int32)
    outputs_array = np.array(outputs_expanded, dtype=np.int32)

    # Initialize feature arrays
    inputs_order2: Optional[NDArray[np.uint8]] = None
    outputs_order2: Optional[NDArray[np.uint8]] = None
    inputs_order3: Optional[NDArray[np.uint8]] = None
    outputs_order3: Optional[NDArray[np.uint8]] = None
    inputs_ncolors: Optional[NDArray[np.uint8]] = None
    outputs_ncolors: Optional[NDArray[np.uint8]] = None
    inputs_is_mask: Optional[NDArray[np.uint8]] = None
    outputs_is_mask: Optional[NDArray[np.uint8]] = None
    inputs_ncells_matching_center: Optional[NDArray[np.uint8]] = None
    outputs_ncells_matching_center: Optional[NDArray[np.uint8]] = None

    if feature_order2:
        print("Computing order-2 (pairwise) relational features...")
        inputs_order2_list: List[NDArray[np.uint8]] = []
        outputs_order2_list: List[NDArray[np.uint8]] = []

        for i in tqdm(
            range(len(inputs_array)), desc="Computing input order-2 features"
        ):
            inputs_order2_list.append(compute_order2_features(inputs_array[i]))

        for i in tqdm(
            range(len(outputs_array)), desc="Computing output order-2 features"
        ):
            outputs_order2_list.append(compute_order2_features(outputs_array[i]))

        inputs_order2 = np.array(inputs_order2_list, dtype=np.uint8)
        outputs_order2 = np.array(outputs_order2_list, dtype=np.uint8)

    if feature_order3:
        print("Computing order-3 (triplet) relational features...")
        inputs_order3_list: List[NDArray[np.uint8]] = []
        outputs_order3_list: List[NDArray[np.uint8]] = []

        for i in tqdm(
            range(len(inputs_array)), desc="Computing input order-3 features"
        ):
            inputs_order3_list.append(compute_order3_features(inputs_array[i]))

        for i in tqdm(
            range(len(outputs_array)), desc="Computing output order-3 features"
        ):
            outputs_order3_list.append(compute_order3_features(outputs_array[i]))

        inputs_order3 = np.array(inputs_order3_list, dtype=np.uint8)
        outputs_order3 = np.array(outputs_order3_list, dtype=np.uint8)

    if feature_ncolors:
        print("Computing number of distinct colors in 3x3 neighborhood...")
        inputs_ncolors_list: List[NDArray[np.uint8]] = []
        outputs_ncolors_list: List[NDArray[np.uint8]] = []

        for i in tqdm(
            range(len(inputs_array)), desc="Computing input number of colors"
        ):
            inputs_ncolors_list.append(compute_ncolors_features(inputs_array[i]))

        for i in tqdm(
            range(len(outputs_array)), desc="Computing output number of colors"
        ):
            outputs_ncolors_list.append(compute_ncolors_features(outputs_array[i]))

        inputs_ncolors = np.array(inputs_ncolors_list, dtype=np.uint8)
        outputs_ncolors = np.array(outputs_ncolors_list, dtype=np.uint8)

    if feature_is_mask:
        print("Computing binary mask feature...")
        inputs_is_mask_list: List[NDArray[np.uint8]] = []
        outputs_is_mask_list: List[NDArray[np.uint8]] = []

        for i in tqdm(range(len(inputs_array)), desc="Computing input mask features"):
            inputs_is_mask_list.append(compute_is_mask_features(inputs_array[i]))

        for i in tqdm(range(len(outputs_array)), desc="Computing output mask features"):
            outputs_is_mask_list.append(compute_is_mask_features(outputs_array[i]))

        inputs_is_mask = np.array(inputs_is_mask_list, dtype=np.uint8)
        outputs_is_mask = np.array(outputs_is_mask_list, dtype=np.uint8)

    if feature_ncells_matching_center:
        print("Computing number of cells matching center cell color...")
        inputs_ncells_matching_center_list: List[NDArray[np.uint8]] = []
        outputs_ncells_matching_center_list: List[NDArray[np.uint8]] = []

        for i in tqdm(
            range(len(inputs_array)), desc="Computing input number of matching cells"
        ):
            inputs_ncells_matching_center_list.append(
                compute_ncells_matching_center_features(inputs_array[i])
            )

        for i in tqdm(
            range(len(outputs_array)), desc="Computing output number of matching cells"
        ):
            outputs_ncells_matching_center_list.append(
                compute_ncells_matching_center_features(outputs_array[i])
            )

        inputs_ncells_matching_center = np.array(
            inputs_ncells_matching_center_list, dtype=np.uint8
        )
        outputs_ncells_matching_center = np.array(
            outputs_ncells_matching_center_list, dtype=np.uint8
        )

    return (
        filenames,
        indices,
        subset_example_index_is_train,
        inputs_array,
        outputs_array,
        inputs_order2,
        outputs_order2,
        inputs_order3,
        outputs_order3,
        inputs_ncolors,
        outputs_ncolors,
        inputs_is_mask,
        outputs_is_mask,
        inputs_ncells_matching_center,
        outputs_ncells_matching_center,
    )


def compute_filename_color_mapping(
    filenames: List[str],
    inputs_array: NDArray[np.int32],
) -> Tuple[NDArray[np.str_], NDArray[np.uint8]]:
    """
    Computes a mapping from unique filenames to the set of colors used across all their examples.

    Args:
        filenames: List of filenames for each example
        inputs_array: numpy array of shape (n_examples, 30, 30) with color values

    Returns:
        filename_colors_keys: numpy array of unique filenames
        filename_colors_values: numpy array of shape (n_unique_filenames, 10) with binary values
                               indicating which colors are used by each filename
    """
    from collections import defaultdict

    # Build mapping from filename to set of colors used
    filename_to_colors: Dict[str, Set[int]] = defaultdict(set)

    for fname, input_grid in zip(filenames, inputs_array):
        # Get all colors in this example (excluding mask -1)
        colors_in_example = set(input_grid.flatten())
        colors_in_example.discard(-1)  # Remove mask

        # Add to the filename's color set
        filename_to_colors[fname].update(colors_in_example)

    # Convert to sorted arrays for consistent ordering
    unique_filenames = sorted(filename_to_colors.keys())

    # Create binary matrix indicating which colors each filename uses
    n_filenames = len(unique_filenames)
    filename_colors_values = np.zeros((n_filenames, 10), dtype=np.uint8)

    for i, fname in enumerate(unique_filenames):
        colors = filename_to_colors[fname]
        for color in colors:
            if 0 <= color <= 9:  # Ensure valid color range
                filename_colors_values[i, color] = 1

    filename_colors_keys = np.array(unique_filenames, dtype=np.str_)

    # Print some statistics
    print(f"\nFilename color mapping statistics:")
    print(f"  Total unique filenames: {n_filenames}")

    # Count how many colors each filename uses
    colors_per_file = filename_colors_values.sum(axis=1)
    for n_colors in range(1, 11):
        count = (colors_per_file == n_colors).sum()
        if count > 0:
            print(f"  Files using {n_colors} colors: {count}")

    return filename_colors_keys, filename_colors_values


def save_data_to_npz(
    directory: str,
    output_path: str,
    filename_filter: Optional[str] = None,
    feature_order2: bool = False,
    feature_order3: bool = False,
    feature_ncolors: bool = False,
    feature_is_mask: bool = False,
    feature_ncells_matching_center: bool = False,
    augment_d4_symmetry: bool = False,
) -> None:
    """
    Loads data using load_data_from_directory and saves it as an NPZ file.
    All inputs and outputs arrays are guaranteed to be (n_examples, 30, 30) int32 arrays.
    All enabled features are concatenated into single inputs_features and outputs_features arrays.

    Args:
        directory: Directory containing the JSON files
        output_path: Path to save the NPZ file
        filename_filter: Optional filter for specific filenames
        feature_order2: Whether to compute and save order-2 relational features (44 features)
        feature_order3: Whether to compute and save order-3 relational features (84 features)
        feature_ncolors: Whether to compute and save number of distinct colors features (9 boolean features)
        feature_is_mask: Whether to compute and save binary mask features (1 feature)
        feature_ncells_matching_center: Whether to compute and save number of matching center cell features (9 boolean features)
        augment_d4_symmetry: Whether to apply all 8 D4 symmetry transformations to augment data (8x increase)

    Saved NPZ file contains:
        - inputs: (n_examples, 30, 30) int32 array with color values
        - outputs: (n_examples, 30, 30) int32 array with color values
        - filenames: array of source JSON filenames
        - indices: array of example indices within each JSON file
        - subset_example_index_is_train: boolean array indicating if example is from "train" (True) or "test" (False) subset
        - inputs_features: (n_examples, 30, 30, n_features) uint8 array with concatenated features (if any features enabled)
        - outputs_features: (n_examples, 30, 30, n_features) uint8 array with concatenated features (if any features enabled)
        - feature_names: array of feature type names (if any features enabled)
    """
    # Load the data
    (
        filenames,
        indices,
        subset_example_index_is_train,
        inputs_array,
        outputs_array,
        inputs_order2,
        outputs_order2,
        inputs_order3,
        outputs_order3,
        inputs_ncolors,
        outputs_ncolors,
        inputs_is_mask,
        outputs_is_mask,
        inputs_ncells_matching_center,
        outputs_ncells_matching_center,
    ) = load_data_from_directory(
        directory,
        filename_filter,
        feature_order2,
        feature_order3,
        feature_ncolors,
        feature_is_mask,
        feature_ncells_matching_center,
        augment_d4_symmetry,
    )

    print(f"Loaded {len(filenames)} examples")
    if augment_d4_symmetry:
        print(f"Note: Data augmented with D4 symmetry (8x multiplication)")
    print(f"Inputs shape: {inputs_array.shape}, dtype: {inputs_array.dtype}")
    print(f"Outputs shape: {outputs_array.shape}, dtype: {outputs_array.dtype}")

    # Count train vs test examples
    train_count = sum(subset_example_index_is_train)
    test_count = len(subset_example_index_is_train) - train_count
    print(f"Examples from 'train' subset: {train_count}")
    print(f"Examples from 'test' subset: {test_count}")

    # Compute filename to color mapping
    filename_colors_keys, filename_colors_values = compute_filename_color_mapping(
        filenames, inputs_array
    )

    inputs_mask = (inputs_array != -1) & (inputs_array != 10)
    outputs_mask = (outputs_array != -1) & (outputs_array != 10)

    # Prepare data to save
    save_data = {
        "inputs": inputs_array,
        "outputs": outputs_array,
        "filenames": np.array(filenames),
        "indices": np.array(indices, dtype=np.int32),
        "subset_example_index_is_train": np.array(
            subset_example_index_is_train, dtype=bool
        ),
        "filename_colors_keys": filename_colors_keys,
        "filename_colors_values": filename_colors_values,
        "inputs_mask": inputs_mask,
        "outputs_mask": outputs_mask,
    }

    # Concatenate all features into single tensors
    inputs_features_list: List[NDArray[np.uint8]] = []
    outputs_features_list: List[NDArray[np.uint8]] = []
    feature_names: List[str] = []

    if feature_order2 and inputs_order2 is not None and outputs_order2 is not None:
        print(f"Adding order2 features: {inputs_order2.shape[-1]} dimensions")
        inputs_features_list.append(inputs_order2)
        outputs_features_list.append(outputs_order2)
        feature_names.append("order2")

    if feature_order3 and inputs_order3 is not None and outputs_order3 is not None:
        print(f"Adding order3 features: {inputs_order3.shape[-1]} dimensions")
        inputs_features_list.append(inputs_order3)
        outputs_features_list.append(outputs_order3)
        feature_names.append("order3")

    if feature_ncolors and inputs_ncolors is not None and outputs_ncolors is not None:
        print(f"Adding ncolors features: {inputs_ncolors.shape[-1]} dimensions")
        inputs_features_list.append(inputs_ncolors)
        outputs_features_list.append(outputs_ncolors)
        feature_names.append("ncolors")

    if feature_is_mask and inputs_is_mask is not None and outputs_is_mask is not None:
        print(f"Adding is_mask features: {inputs_is_mask.shape[-1]} dimensions")
        inputs_features_list.append(inputs_is_mask)
        outputs_features_list.append(outputs_is_mask)
        feature_names.append("is_mask")

    if (
        feature_ncells_matching_center
        and inputs_ncells_matching_center is not None
        and outputs_ncells_matching_center is not None
    ):
        print(
            f"Adding ncells_matching_center features: {inputs_ncells_matching_center.shape[-1]} dimensions"
        )
        inputs_features_list.append(inputs_ncells_matching_center)
        outputs_features_list.append(outputs_ncells_matching_center)
        feature_names.append("ncells_matching_center")

    # Concatenate features if any are present
    inputs_features: Optional[NDArray[np.uint8]] = None
    outputs_features: Optional[NDArray[np.uint8]] = None

    if inputs_features_list:
        inputs_features = np.concatenate(inputs_features_list, axis=-1)
        outputs_features = np.concatenate(outputs_features_list, axis=-1)

        print(f"Combined features: {', '.join(feature_names)}")
        print(f"Total feature dimensions: {inputs_features.shape[-1]}")
        print(
            f"Inputs features shape: {inputs_features.shape}, dtype: {inputs_features.dtype}"
        )
        print(
            f"Outputs features shape: {outputs_features.shape}, dtype: {outputs_features.dtype}"
        )

        save_data["inputs_features"] = inputs_features
        save_data["outputs_features"] = outputs_features
        save_data["feature_names"] = np.array(
            feature_names, dtype="U32"
        )  # Save feature names for reference
    else:
        print("No features enabled")

    # Save as NPZ file with all data
    np.savez_compressed(output_path, **save_data)  # type: ignore
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Preprocess ARC-AGI data and save to NPZ files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ARC-AGI/data",
        help="Directory containing the ARC-AGI data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Directory to save the NPZ files",
    )
    parser.add_argument(
        "--feature_order2",
        action="store_true",
        help="Compute order-2 (pairwise) relational features (adds 44-feature arrays)",
        default=False,
    )
    parser.add_argument(
        "--feature_order3",
        action="store_true",
        help="Compute order-3 (triplet) relational features (adds 84-feature arrays)",
        default=False,
    )
    parser.add_argument(
        "--feature_ncolors",
        action="store_true",
        help="Compute number of distinct colors in 3x3 neighborhood (adds 1-feature arrays)",
        default=False,
    )
    parser.add_argument(
        "--feature_is_mask",
        action="store_true",
        help="Compute binary mask feature (adds 1-feature arrays)",
        default=False,
    )
    parser.add_argument(
        "--feature_ncells_matching_center",
        action="store_true",
        help="Compute number of cells matching center cell color (adds 1-feature arrays)",
        default=False,
    )
    parser.add_argument(
        "--feature_all",
        action="store_true",
        help="Compute all available features (equivalent to enabling all feature flags)",
        default=False,
    )
    parser.add_argument(
        "--augment_d4_symmetry",
        action="store_true",
        help="Apply all 8 D4 symmetry transformations to augment data (8x increase in dataset size)",
        default=True,
    )
    args = parser.parse_args()

    # If feature_all is enabled, turn on all individual features
    if args.feature_all:
        args.feature_order2 = True
        args.feature_order3 = True
        args.feature_ncolors = True
        args.feature_is_mask = True
        args.feature_ncells_matching_center = True

    # Create paths
    data_dir: Path = Path(args.data_dir)
    output_dir: Path = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths for training and evaluation data
    train_dir: str = str(data_dir / "training")
    eval_dir: str = str(data_dir / "evaluation")

    # Create output filenames based on which features are requested
    suffix = ""
    if args.feature_all:
        suffix = "_all"
    else:
        if args.feature_order2:
            suffix += "_order2"
        if args.feature_order3:
            suffix += "_order3"
        if args.feature_ncolors:
            suffix += "_ncolors"
        if args.feature_is_mask:
            suffix += "_is_mask"
        if args.feature_ncells_matching_center:
            suffix += "_ncells_matching_center"
        if suffix == "":
            suffix = ""  # No features requested, use default names

    # Add augmentation suffix if enabled
    if args.augment_d4_symmetry:
        suffix += "_d4aug"

    train_output: str = str(output_dir / f"train{suffix}.npz")
    eval_output: str = str(output_dir / f"eval{suffix}.npz")

    # Check if directories exist
    if Path(train_dir).exists():
        print(f"Processing training data from {train_dir}")
        print(f"Saving to {train_output}")
        save_data_to_npz(
            train_dir,
            train_output,
            feature_order2=args.feature_order2,
            feature_order3=args.feature_order3,
            feature_ncolors=args.feature_ncolors,
            feature_is_mask=args.feature_is_mask,
            feature_ncells_matching_center=args.feature_ncells_matching_center,
            augment_d4_symmetry=args.augment_d4_symmetry,
        )
        print("Training data saved successfully")
    else:
        print(f"Training directory {train_dir} does not exist, skipping...")

    if Path(eval_dir).exists():
        print(f"Processing evaluation data from {eval_dir}")
        print(f"Saving to {eval_output}")
        save_data_to_npz(
            eval_dir,
            eval_output,
            feature_order2=args.feature_order2,
            feature_order3=args.feature_order3,
            feature_ncolors=args.feature_ncolors,
            feature_is_mask=args.feature_is_mask,
            feature_ncells_matching_center=args.feature_ncells_matching_center,
            augment_d4_symmetry=args.augment_d4_symmetry,
        )
        print("Evaluation data saved successfully")
    else:
        print(f"Evaluation directory {eval_dir} does not exist, skipping...")
