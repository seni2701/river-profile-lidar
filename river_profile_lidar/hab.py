import geopandas
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from river_profile_lidar.bathymetry.hab.create_hab_shade import (
    process_and_save_hab,
)

log = logging.getLogger(__name__)

APPLY_SOBEL = True
RASTER_RESOLUTION = 0.3
MIN_PROPORTION_OF_NOT_IN_SHADE_PIXELS = 0.4


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing HAB for: {cfg.river.name}")

    rgb_image_folder_path = cfg.river.rgb_image_folder_path
    transect_path = cfg.river.transect_path
    index_file_path = cfg.river.index_file_path
    trapezoid_folder_path = cfg.river.trapezoid_output_path
    saving_folder_path = Path(cfg.river.hab_output_path)

    shade_kwargs = {
        "shade_mask_path": cfg.river.shade_mask_path,
        "cross_section_points_path": Path(
            cfg.river.cross_section_output_path, "hab_cross_section_points.shp"
        ),
        "min_proportion_of_not_in_shade_pixels": MIN_PROPORTION_OF_NOT_IN_SHADE_PIXELS,
    }

    image_to_process_df = geopandas.read_file(index_file_path)

    hierarchie_col_name = (
        "Hierarchie" if "Hierarchie" in image_to_process_df.columns else "HIERARCHIE"
    )
    image_to_process_df = image_to_process_df.sort_values(
        by=hierarchie_col_name
    ).reset_index(drop=True)

    # ✅ CORRECTION CLÉ : on conserve le nom réel des images
    image_to_process_df["nom_image_save"] = (
        image_to_process_df["NOM_IMAGE"]
    )

    image_name_to_process_list = image_to_process_df["nom_image_save"].to_list()
    number_image_to_process = len(image_name_to_process_list)

    for i, image_name in enumerate(image_name_to_process_list):
        log.info(f"image: {image_name} {i+1}/{number_image_to_process}")

        output_hab = Path(
            saving_folder_path, f"{Path(image_name).stem}_hab.tif"
        )
        if output_hab.exists():
            continue

        rgb_image_path = Path(rgb_image_folder_path, image_name + ".tif")
        if not rgb_image_path.exists():
            log.warning(f"Image introuvable : {rgb_image_path}")
            continue

        shade_kwargs["trapezoid_path"] = Path(
            trapezoid_folder_path, f"{Path(image_name).stem}_trapezoid.tif"
        )

        saving_folder_path.mkdir(parents=True, exist_ok=True)
        process_and_save_hab(
            rgb_image_path=rgb_image_path,
            transect_path=transect_path,
            apply_sobel=APPLY_SOBEL,
            shade_kwargs=shade_kwargs,
            raster_resolution=RASTER_RESOLUTION,
            output_path=saving_folder_path,
        )

        log.info(f"image: {image_name} Is Done")


if __name__ == "__main__":
    run()
