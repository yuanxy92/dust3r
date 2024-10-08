import os
import shutil

def merge_data_dirs(source_dirs, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    image_count = 0

    for source_dir in source_dirs:
        for i in range(4):  # There are 4 images in each dir: 0.png, 1.png, 2.png, 3.png
            source_image = os.path.join(source_dir, f"{i}.png")
            if os.path.exists(source_image):
                target_image = os.path.join(target_dir, f"image_{image_count}.png")
                shutil.copy(source_image, target_image)
                image_count += 1
            else:
                print(f"Image {i}.png not found in {source_dir}")

if __name__ == "__main__":
    # rootdir = '/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241008_1'
    # # Example usage
    # source_dirs = [f'{rootdir}/005', f'{rootdir}/006']  # List of source directories
    # target_dir = f'{rootdir}/014'  # Target directory to store all merged images

    rootdir = '/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4'
    # Example usage
    source_dirs = [f'{rootdir}/004', f'{rootdir}/009']  # List of source directories
    target_dir = f'{rootdir}/024'  # Target directory to store all merged images

    merge_data_dirs(source_dirs, target_dir)
    print(f"All images have been copied to {target_dir}")
