from pathlib import Path

from nii_2_precomputed import Resolution, convert_nii_to_precomputed

image_path = Path(r"D:\EEG Data\nii\20230608\stitched_0004.nii.gz")
base_dir = Path(r"D:\EEG Data\nii")

folder_name = image_path.name
if (i := folder_name.find(".nii")) >= 0:
    folder_name = folder_name[:i]
out_folder = image_path.parent / folder_name

out_folder_url_path = "/".join(out_folder.relative_to(base_dir).parts)
url_path = f"http://localhost:8080/{out_folder_url_path}"

convert_nii_to_precomputed(
    out_folder, image_path, url_path, resolution=Resolution(10_000, 10_000, 10_000)
)
