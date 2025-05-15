import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore

def load_and_resize_pattern(pattern_path, target_shape):
    pattern = cv2.imread(pattern_path)
    return cv2.resize(pattern, (target_shape[1], target_shape[0]))

def get_tshirt_mask(image, pose_landmarks):
    height, width, _ = image.shape

    # Get coordinates for shoulders and hips
    left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

    # Convert to pixel coordinates
    points = np.array([
        [int(left_shoulder.x * width), int(left_shoulder.y * height)],
        [int(right_shoulder.x * width), int(right_shoulder.y * height)],
        [int(right_hip.x * width), int(right_hip.y * height)],
        [int(left_hip.x * width), int(left_hip.y * height)]
    ], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    return mask

def overlay_tshirt(image_path, pattern_path):
    image = cv2.imread(image_path)
    pattern = load_and_resize_pattern(pattern_path, image.shape)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            print("Pose not detected.")
            return image

        tshirt_mask = get_tshirt_mask(image, result.pose_landmarks)

        # Optional: Smooth mask
        tshirt_mask = cv2.GaussianBlur(tshirt_mask, (15, 15), 0)
        mask_inv = cv2.bitwise_not(tshirt_mask)

        pattern_on_body = cv2.bitwise_and(pattern, pattern, mask=tshirt_mask)
        background = cv2.bitwise_and(image, image, mask=mask_inv)
        final = cv2.add(background, pattern_on_body)

        return final

# MAIN
image_path = 'person.png'      # Your person image
pattern_path = 'shirt.png'     # Pattern or t-shirt

output = overlay_tshirt(image_path, pattern_path)
cv2.imshow("Virtual T-Shirt Try-On", output)
cv2.imwrite("virtual_tshirt_result.jpg", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
