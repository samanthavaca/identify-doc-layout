import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

# --- Python functions for feature matching ---

# Calculate the median angle from the keypoint orientations. Using the median seems to work as a healthier average for the document angle.
def calculate_median_angle(good_matches, keypoints_template, keypoints_image):
    angles = []
    for match in good_matches:
        kp_template = keypoints_template[match.queryIdx]
        kp_image = keypoints_image[match.trainIdx]
        
        # Difference in keypoint orientations
        angle_diff = kp_image.angle - kp_template.angle
        angles.append(angle_diff)
    
    median_angle = np.median(angles)
    return median_angle

# This function calculates the rotation angle from the homography matrix
def extract_rotation_angle(M):
    r_11, r_12 = M[0, 0], M[0, 1]
    r_21, r_22 = M[1, 0], M[1, 1]

    theta = np.arctan2(r_21, r_11)

    angle = np.degrees(theta)

    return angle

def safely_get_rotation_matrix(cx, cy, angle):
    # If the angle is very small, don't apply a rotation
    if abs(angle) < 1.0:
        print(f"Skipping rotation for small angle: {angle}")
        return np.eye(2, 3)
    return cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

# Show the bounding box on a grid!
def plot_bounding_box_with_grid(image, dst, bounding_box):
    plt.figure(figsize=(10, 10))

    # Plot the input image
    plt.imshow(image, cmap='gray')

    # Draw the bounding box by drawing each line
    bounding_box = np.int32(bounding_box)
    plt.plot([bounding_box[0][0][0], bounding_box[1][0][0]], [bounding_box[0][0][1], bounding_box[1][0][1]], color='red', linewidth=2) 
    plt.plot([bounding_box[1][0][0], bounding_box[2][0][0]], [bounding_box[1][0][1], bounding_box[2][0][1]], color='red', linewidth=2)
    plt.plot([bounding_box[2][0][0], bounding_box[3][0][0]], [bounding_box[2][0][1], bounding_box[3][0][1]], color='red', linewidth=2) 
    plt.plot([bounding_box[3][0][0], bounding_box[0][0][0]], [bounding_box[3][0][1], bounding_box[0][0][1]], color='red', linewidth=2)

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.title("Image with Bounding Box")
    plt.show()

def is_bad_crop(image, min_content_ratio=0.15):
    """
    Check if the content in the image is too sparse.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    content_area = np.count_nonzero(binary)
    total_area = image.shape[0] * image.shape[1]
    content_ratio = content_area / total_area

    if content_ratio < min_content_ratio:
        return 0.0  # Very bad crop: Too little content
    elif content_ratio < min_content_ratio * 2:
        return 0.3  # Poor crop
    return 1.0  # Good content ratio

def measure_blurriness(image, blur_threshold=30.0):
    """
    Measures the sharpness of the image using Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < blur_threshold / 2:
        return 0.0  # Very blurry
    elif laplacian_var < blur_threshold:
        return 0.5  # Slightly blurry
    return 1.0  # Sharp image

def size_ratio_check(original_image, cropped_image, min_ratio=0.1):
    """
    Penalizes crops that are excessively small compared to the original image.
    """
    original_area = original_image.shape[0] * original_image.shape[1]
    cropped_area = cropped_image.shape[0] * cropped_image.shape[1]
    ratio = cropped_area / original_area

    if ratio < min_ratio:
        return 0.0  # Excessively zoomed-in crop
    elif ratio < min_ratio * 2:
        return 0.3  # Small crop
    return 1.0  # Reasonable size

def centrality_check(original_image, cropped_image):
    """
    Ensures the cropped area is not zoomed too close to the edges.
    """
    original_h, original_w = original_image.shape[:2]
    cropped_h, cropped_w = cropped_image.shape[:2]

    # Calculate the center of both images
    orig_center = (original_w // 2, original_h // 2)
    crop_center = (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)

    # Compute distance from crop center to original center
    dist_x = abs(orig_center[0] - crop_center[0])
    dist_y = abs(orig_center[1] - crop_center[1])

    max_allowed_dist = 0.25 * min(original_w, original_h)  # Allow 25% offset
    if dist_x > max_allowed_dist or dist_y > max_allowed_dist:
        return 0.3  # Crop is too far off-center
    return 1.0  # Centered crop

def keyword_detection(image, keywords):
    """
    Detects essential keywords using OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    text = pytesseract.image_to_string(gray).lower()

    missing_count = sum(1 for kw in keywords if kw.lower() not in text)
    if missing_count > len(keywords) / 2:
        return 0.0  # Most keywords are missing
    elif missing_count > 0:
        return 0.5  # Some keywords are missing
    return 1.0  # All essential keywords found

def check_cropped_image_quality(original_image, cropped_image, keywords=[]):
    """
    Computes the overall quality score of the cropped image.
    """
    scores = []

    # Content Area Ratio Score
    scores.append(is_bad_crop(cropped_image))

    # Size Ratio Score
    scores.append(size_ratio_check(original_image, cropped_image))

    # Centrality Check Score
    scores.append(centrality_check(original_image, cropped_image))

    # Blur Score
    scores.append(measure_blurriness(cropped_image))

    # Keyword Detection Score (if keywords are provided)
    if keywords:
        scores.append(keyword_detection(cropped_image, keywords))

    # Compute the weighted geometric mean of the scores
    overall_score = (np.prod([max(0.1, s) for s in scores]) ** (1 / len(scores))) * 100
    return overall_score

# Draw the matched features between the input and the template
def draw_matched_features(template, image, keypoints_template, keypoints_image, good_matches):
    matched_image = cv2.drawMatches(
        template, keypoints_template, image, keypoints_image, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(16, 12))
    plt.imshow(matched_image)
    plt.title('Matched Keypoints')
    plt.show()

def identify_doc_layout(template, image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and get the descriptions of the keypoints for each image
    keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
    keypoints_image, descriptors_image = sift.detectAndCompute(image, None)
    
    # Cross-check Brute-Force matcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Get the matches
    good_matches = bf.match(descriptors_template, descriptors_image)
    
    # Sort the matches
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # Check if enough matches are found. The threshold is currently random, but I just didn't want it to proceed with barely any matches.
    if len(good_matches) > 800:
        
        # Draw the matched features and show them
        draw_matched_features(template, image, keypoints_template, keypoints_image, good_matches)
        
        # Get the keypoints of the good matches for both the template and input images
        src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if M is None:
            print("Homography matrix is invalid.")
            return None

        if np.allclose(M, np.eye(3), atol=1e-2):
            print("Homography matrix is too close to identity, skipping transformation.")
            return None

        # Get the bounding box from the template
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # Transform the bounding box to the region in the image that matches the template
        dst = cv2.perspectiveTransform(pts, M)

        # Ensure that all transformed points are within the image boundaries
        dst[:, :, 0] = np.clip(dst[:, :, 0], 0, image.shape[1] - 1)
        dst[:, :, 1] = np.clip(dst[:, :, 1], 0, image.shape[0] - 1)

        # Old: draw bounding box 
        image_with_box = cv2.polylines(image.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        
        plot_bounding_box_with_grid(image, dst, dst)

        # Crop the image to include only the bounding box
        x, y, w, h = cv2.boundingRect(dst)
        print(f"Bounding Box Coordinates: x={x}, y={y}, w={w}, h={h}")
        print(f"Image Shape: {image.shape}")
        cropped_image = image[y:y+h, x:x+w]
        
        # Get the bounding box and angle
        rect = cv2.minAreaRect(dst)
        angle = rect[-1]
        print(angle)
        angle = extract_rotation_angle(M)
        print(f"Rotation Angle: {angle}")
        angle = calculate_median_angle(good_matches, keypoints_template, keypoints_image)
        print(f"Median Angle: {angle}")
        
        # Calculate the image center using the bounding box
        h, w = cropped_image.shape[:2]
        cx, cy = (w // 2, h // 2)

        # Get the rotation matrix
        M = safely_get_rotation_matrix(cx, cy, angle)

        # Get the cosine and sine values of the rotation
        cos, sin = abs(M[0, 0]), abs(M[0, 1])

        # Calulate the new width and height from the resulting rotation
        newW = int((h * sin) + (w * cos))
        newH = int((h * cos) + (w * sin))

        # Calculate new rotation center
        M[0, 2] += (newW / 2) - cx
        M[1, 2] += (newH / 2) - cy

        # Use warpAffine method to rotate the cropped image
        result = cv2.warpAffine(cropped_image, M, (newW, newH), flags=cv2.INTER_LINEAR) 
        
        # Display the cropped region
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))
        plt.title('Cropped Image')
        plt.axis('off')
        plt.show()

        return result

    else:
        print("Not enough matches are found to be able to crop - {}/{}. Please make sure that all documents are of the same type.".format(len(good_matches), 0))
        return None

# --- Streamlit ---
st.title("Feature Matching for Document Layout Identification")

template_file = st.file_uploader("Upload Template Image", type=["jpg", "jpeg", "png"])
input_file = st.file_uploader("Upload Input Image", type=["jpg", "jpeg", "png"])

if template_file and input_file:
    template = np.array(Image.open(template_file).convert("L"))
    input_image = np.array(Image.open(input_file).convert("L"))

    cropped_image = identify_doc_layout(template, input_image)

    if cropped_image is not None:
        quality_score = check_cropped_image_quality(input_image, cropped_image)

        if quality_score < 50:
            st.error("The cropped image quality is poor. Please make sure you submitted images with documents of the same type.")
        else:
            st.success(f"Cropped Image Quality Score: {quality_score:.2f}")
            st.image(cropped_image, caption="Cropped Image", use_column_width=True)
    else:
        st.error("Failed to crop the document.")
