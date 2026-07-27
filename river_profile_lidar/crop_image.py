"""
Crop images according to an index file that contains a HIERARCHIE column.

Rule:
- Smaller hierarchy value = higher priority (1 > 2 > 3 > ...)
- If two images INTERSECT, the image with the LARGER hierarchy value is cropped.
"""
import geopandas
from pathlib import Path
from tqdm import tqdm
import logging
import hydra
from omegaconf import DictConfig

from river_profile_lidar.crop_image_tools.utils import crop_image_under_from_2_images

log = logging.getLogger(__name__)

DELETE_UNUSED_IMAGE_IN_FOLDER = False


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing Crop Image for: {cfg.river.name}")

    index_file_path = cfg.river.index_file_path
    image_folder_path = cfg.river.rgb_image_folder_path

    image_to_process_df = geopandas.read_file(index_file_path)

    # Priorité : plus petite valeur = plus prioritaire
    image_to_process_df = image_to_process_df.sort_values(
        "Hierarchie"
    ).reset_index(drop=True)

    # Conserver le nom exact de l’image
    image_to_process_df["nom_image_save"] = (
        image_to_process_df["NOM_IMAGE"] + ".tif"
    )

    if DELETE_UNUSED_IMAGE_IN_FOLDER:
        for image_path in filter(
            lambda p: p.suffix.lower() == ".tif",
            Path(image_folder_path).iterdir(),
        ):
            if image_path.name not in image_to_process_df["nom_image_save"].values:
                image_path.unlink()

    for low_idx, low_row in tqdm(
        image_to_process_df[:-1].iterrows(),
        total=len(image_to_process_df[:-1]),
    ):
        # Image prioritaire (hiérarchie plus petite)
        image_low_path = Path(image_folder_path, low_row["nom_image_save"])

        if not image_low_path.exists():
            log.warning(f"Image introuvable : {image_low_path}")
            continue

        for _, high_row in image_to_process_df[low_idx + 1 :].iterrows():
            # Image moins prioritaire (hiérarchie plus grande)
            image_high_path = Path(image_folder_path, high_row["nom_image_save"])

            if not image_high_path.exists():
                log.warning(f"Image introuvable : {image_high_path}")
                continue

            # 🔴 CORRECTION CLÉ ICI
            if low_row.geometry.intersects(high_row.geometry):
                log.info(
                    f"CROP: H={high_row['Hierarchie']} -> "
                    f"H={low_row['Hierarchie']}"
                )

                # On découpe UNE seule image : la moins prioritaire
                crop_image_under_from_2_images(
                    image_high_path,  # coupée
                    image_low_path,   # conservée
                )


if __name__ == "__main__":
    run()
