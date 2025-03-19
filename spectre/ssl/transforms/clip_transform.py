from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
)

from spectre.transforms import SWSpatialCropSamplesd


class CLIPTransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
        ):
        super().__init__(
            [
                LoadImaged(keys=("image",)),
                EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=("image",), 
                    a_min=-1000, 
                    a_max=1000, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True
                ),
                Orientationd(keys=("image",), axcodes="RAS"),
                Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
                ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, 256)),
                
                # Crop the volume into equal non-overlapping samples
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                )
            ]
        )


if __name__ == "__main__":

    # Save some example data after transforming it.
    import os
    import SimpleITK as sitk

    data = {"image": r"data/test_data/train_1_a_1.nii.gz"}
    transform = CLIPTransform()
    transformed_data = transform(data)

    # Save the different crops to a folder for visualization.
    output_dir = r"data/test_data/clip_transform_output"
    os.makedirs(output_dir, exist_ok=True)

    for i, patch in enumerate(transformed_data):

        # Save the crops
        patch_img = sitk.GetImageFromArray(patch["image"].squeeze(0).numpy())
        patch_img.SetSpacing((1.5, 0.75, 0.75))
        patch_path = os.path.join(output_dir, f"{i}_crop.nii.gz")
        sitk.WriteImage(patch_img, patch_path)
