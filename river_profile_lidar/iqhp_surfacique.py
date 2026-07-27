import geopandas
from pathlib import Path
import logging

from habitat_model import HabitatModel  # noqa: F401
from river_profile_lidar.iqh.create_iqh import process_and_save_iqh
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

RASTER_RESOLUTION = 0.3


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing IQH for: {cfg.river.name}")

    hab_image_folder_path = Path(cfg.river.hab_output_path)
    water_speed_image_folder_path = Path(cfg.river.water_speed_output_path)
    d84_image_folder_path = Path(cfg.river.d84_output_path)
    index_file_path = Path(cfg.river.index_file_path)
    saving_folder_path = Path(cfg.river.iqh_output_path)

    image_to_process_df = geopandas.read_file(index_file_path)

    hierarchie_col_name = (
        "Hierarchie" if "Hierarchie" in image_to_process_df.columns else "HIERARCHIE"
    )

    image_to_process_df = image_to_process_df.sort_values(
        by=hierarchie_col_name
    ).reset_index(drop=True)

    # ✅ CORRECTION CLÉ : conserver le nom réel des images
    image_to_process_df["nom_image_save"] = (
        image_to_process_df["NOM_IMAGE"]
    )

    image_name_to_process_list = image_to_process_df["nom_image_save"].to_list()
    number_image_to_process = len(image_name_to_process_list)

    for i, image_name in enumerate(image_name_to_process_list):
        image_name_stem = Path(image_name).stem
        log.info(f"image: {image_name} {i+1}/{number_image_to_process}")

        # sortie IQH déjà existante
        if Path(saving_folder_path, f"{image_name_stem}_iqh.tif").exists():
            continue

        # dépendances obligatoires
        hab_image_path = Path(hab_image_folder_path, f"{image_name_stem}_hab.tif")
        water_speed_image_path = Path(
            water_speed_image_folder_path, f"{image_name_stem}_water_speed.tif"
        )
        d84_image_path = Path(d84_image_folder_path, f"{image_name_stem}_d84.tif")

        if not hab_image_path.exists():
            log.warning(f"HAB manquant pour : {image_name_stem}")
            continue
        if not water_speed_image_path.exists():
            log.warning(f"Water speed manquant pour : {image_name_stem}")
            continue
        if not d84_image_path.exists():
            log.warning(f"D84 manquant pour : {image_name_stem}")
            continue

        process_and_save_iqh(
            image_prefix_name=image_name_stem,
            hab_image_path=hab_image_path,
            water_speed_image_path=water_speed_image_path,
            d84_image_path=d84_image_path,
            output_path=saving_folder_path,
        )

        log.info(f"image: {image_name} Is Done")


if __name__ == "__main__":
    run()
