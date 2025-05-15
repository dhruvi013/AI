import cv2  # type: ignore
import numpy as np  # type: ignore

def resize_pattern_to_image(pattern_img, image_shape):
    return cv2.resize(pattern_img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)

def overlay_tshirt_on_green(image, pattern_img):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for green color
    lower_green = np.array([35, 60, 60])  # You can adjust if needed
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_inv = cv2.bitwise_not(mask)

    # Resize pattern to match the image size
    pattern_resized = resize_pattern_to_image(pattern_img, image.shape)

    # Apply the pattern where green is detected
    tshirt_texture = cv2.bitwise_and(pattern_resized, pattern_resized, mask=mask)
    background = cv2.bitwise_and(image, image, mask=mask_inv)

    return cv2.add(background, tshirt_texture)

def main():
    image = cv2.imread('person.png')      # The input image with green shirt
    pattern_img = cv2.imread('image.png') # The pattern to overlay

    if image is None:
        print("Error: 'person.jpg' not found.")
        return
    if pattern_img is None:
        print("Error: 'image.png' not found.")
        return

    result = overlay_tshirt_on_green(image, pattern_img)

    # Show and save the result
    cv2.imshow("Virtual Try-On Result", result)
    cv2.imwrite("result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()