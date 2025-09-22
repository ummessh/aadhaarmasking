import cv2
import numpy as np
import os

def augment_image(image, angle, blur_kernel_size, brightness, contrast):
    # Rotation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Blur
    blurred = cv2.GaussianBlur(rotated, (blur_kernel_size, blur_kernel_size), 0)

    # Brightness and Contrast
    adjusted = cv2.convertScaleAbs(blurred, alpha=contrast, beta=brightness)

    return adjusted

def main():
    template_dir = 'templates'
    output_dir = 'templates'  # Save in the same directory

    if not os.path.isdir(template_dir):
        print(f"Error: Template directory not found at '{template_dir}'")
        return

    # Augmentation parameters
    augmentations = {
        'rotated_5': {'angle': 5, 'blur': 3, 'brightness': 0, 'contrast': 1.0},
        'rotated_neg_5': {'angle': -5, 'blur': 3, 'brightness': 0, 'contrast': 1.0},
        'blurred': {'angle': 0, 'blur': 7, 'brightness': 0, 'contrast': 1.0},
        'bright': {'angle': 0, 'blur': 3, 'brightness': 20, 'contrast': 1.0},
        'dark': {'angle': 0, 'blur': 3, 'brightness': -20, 'contrast': 1.0},
        'high_contrast': {'angle': 0, 'blur': 3, 'brightness': 0, 'contrast': 1.5},
    }

    original_files = [f for f in os.listdir(template_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not any(aug_name in f for aug_name in augmentations.keys())]

    for filename in original_files:
        path = os.path.join(template_dir, filename)
        original_image = cv2.imread(path)

        if original_image is None:
            print(f"Warning: Could not read image {filename}")
            continue

        base_name, extension = os.path.splitext(filename)

        for aug_name, params in augmentations.items():
            augmented_image = augment_image(
                original_image,
                params['angle'],
                params['blur'],
                params['brightness'],
                params['contrast']
            )
            
            new_filename = f"{base_name}_{aug_name}{extension}"
            new_path = os.path.join(output_dir, new_filename)
            cv2.imwrite(new_path, augmented_image)
            print(f"Saved augmented template: {new_filename}")

if __name__ == "__main__":
    main()
