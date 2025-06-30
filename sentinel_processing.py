import ee
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from skimage import filters
import geopandas as gpd
import json
import os

def main():
    # Authentification Earth Engine
    try:
        ee.Initialize(project='tidy-bindery-461215-i7')
    except:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize(project='tidy-bindery-461215-i7')
    
    # Chargement du shapefile
    shapefile_path = "POI_Combe_Valtin.shp"
    gdf = gpd.read_file(shapefile_path)
    if gdf.geometry.is_empty.any():
        raise ValueError("Le shapefile contient des géométries vides")
    geojson = json.loads(gdf.to_json())
    features = []
    for feature in geojson['features']:
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            for ring in geom['coordinates']:
                if len(ring) < 3:
                    raise ValueError("Un polygone nécessite au moins 3 points")
        elif geom['type'] == 'MultiPolygon':
            for polygon in geom['coordinates']:
                for ring in polygon:
                    if len(ring) < 3:
                        raise ValueError("Un polygone nécessite au moins 3 points")
        ee_geom = ee.Geometry(geom)
        features.append(ee.Feature(ee_geom))
    combe_valtin_fc = ee.FeatureCollection(features)
    zones = {
        "combe_valtin": {
            "nom": "Combe Valtin (Shapefile)",
            "roi": combe_valtin_fc.geometry(),
        }
    }
    DATE_DEBUT = '2023-01-01'
    DATE_FIN = '2025-06-30'
    MAX_NUAGES = 60
    BANDES = ['B2', 'B3', 'B4']  # Seulement les bandes visibles

    for zone_id, zone in zones.items():
        print(f"\n=== Traitement de {zone['nom']} ===")
        base_dir = Path('forets_comparaison') / zone_id
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / 'brut').mkdir(exist_ok=True)
        (base_dir / 'SEN2SR').mkdir(exist_ok=True)
        (base_dir / 'comparaisons').mkdir(exist_ok=True)
        print("Recherche d'images Sentinel-2...")
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(zone['roi'])
                      .filterDate(DATE_DEBUT, DATE_FIN)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUAGES)))
        if collection.size().getInfo() == 0:
            print("Aucune image trouvée - Élargissement des critères")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(zone['roi'])
                          .filterDate('2016-01-01', '2018-12-31')
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUAGES)))
        try:
            sorted_collection = collection.sort('CLOUDY_PIXEL_PERCENTAGE')
            image_list = sorted_collection.toList(sorted_collection.size())
            n_images = image_list.size().getInfo()
        except ee.EEException as e:
            print(f"Erreur EE: {e}")
            n_images = collection.size().getInfo()
            image_list = collection.toList(n_images)
        print(f"{n_images} images trouvées pour cette zone")
        for j in range(n_images):
            image = ee.Image(image_list.get(j))
            image_id = image.get('system:id').getInfo()
            cloud_percent = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            print(f"\nTraitement de l'image {j+1}/{n_images} (ID: {image_id}, Nuages: {cloud_percent}%)")
            url = image.getDownloadURL({
                'bands': BANDES,
                'region': zone['roi'],
                'scale': 10,
                'format': 'GEO_TIFF',
                'crs': 'EPSG:32631'
            })
            raw_path = base_dir / 'brut' / f"raw_{image_id.replace('/', '_')}_{j}.tif"
            response = requests.get(url)
            with open(raw_path, 'wb') as f:
                f.write(response.content)
            print(f"Image brute téléchargée: {raw_path}")
            # === 3. COMPOSITIONS DE BASE ===
            with rasterio.open(raw_path) as src:
                data = src.read()
                band_names = ['B2', 'B3', 'B4']  # Seulement les bandes visibles
                bands = {band: data[i] for i, band in enumerate(band_names)}
                def normalize(band):
                    return (band - np.min(band)) / (np.max(band) - np.min(band) + 1e-10)
                # Calcul et sauvegarde seulement RGB
                compositions = {
                    'rgb': np.stack([normalize(bands['B4']), normalize(bands['B3']), normalize(bands['B2'])], axis=-1),
                    # 'ndvi': (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'] + 1e-10),  # COMMENTÉ
                    # 'false_color': np.stack([normalize(bands['B5']), normalize(bands['B8']), normalize(bands['B11'])], axis=-1)  # COMMENTÉ
                }
                for comp_name, comp_data in compositions.items():
                    # if comp_name == 'ndvi':
                    #     plt.imsave(base_dir / 'brut' / f'{comp_name}_{j}.png', comp_data, cmap='RdYlGn', vmin=-1, vmax=1)
                    # else:
                    plt.imsave(base_dir / 'brut' / f'{comp_name}_{j}.png', comp_data)
            # === 4. SIMULATION UNIQUEMENT DE SEN2SR ===
            def simulate_band(band_data, factor, sharpness=True):
                resized = resize(band_data, 
                                (band_data.shape[0] * factor, 
                                 band_data.shape[1] * factor), 
                                anti_aliasing=True)
                if sharpness:
                    return filters.unsharp_mask(resized, radius=2, amount=1.5)
                return resized
            method_params = {
                'SEN2SR': {'factor': 4, 'sharpness': False}
            }
            required_bands = ['B2', 'B3', 'B4']  # Seulement les bandes visibles
            for method, params in method_params.items():
                method_dir = base_dir / method
                print(f"Traitement {method}...")
                sim_bands = {}
                for band_name in required_bands:
                    sim_bands[band_name] = simulate_band(bands[band_name], 
                                                        params['factor'], 
                                                        params['sharpness'])
                comps = {
                    'rgb': np.stack([normalize(sim_bands['B4']), 
                                    normalize(sim_bands['B3']), 
                                    normalize(sim_bands['B2'])], axis=-1),
                    # 'ndvi': (sim_bands['B8'] - sim_bands['B4']) / (sim_bands['B8'] + sim_bands['B4'] + 1e-10),  # COMMENTÉ
                    # 'false_color': np.stack([normalize(sim_bands['B5']), 
                    #                        normalize(sim_bands['B8']), 
                    #                        normalize(sim_bands['B11'])], axis=-1)  # COMMENTÉ
                }
                for comp_name, comp_data in comps.items():
                    # if comp_name == 'ndvi':
                    #     plt.imsave(method_dir / f'{comp_name}_{j}.png', comp_data, cmap='RdYlGn', vmin=-1, vmax=1)
                    # else:
                    plt.imsave(method_dir / f'{comp_name}_{j}.png', comp_data)
            # === 5. COMPARAISONS BRUT vs SEN2SR ===
            methods = ['brut', 'SEN2SR']
            compositions = ['rgb']  # Seulement RGB
            for comp_name in compositions:
                fig, axs = plt.subplots(1, 2, figsize=(13, 6))
                fig.suptitle(f"Comparaison {comp_name} - {zone['nom']} - Image {j+1}", fontsize=16)
                for i, method in enumerate(methods):
                    if method == 'brut':
                        img_path = base_dir / 'brut' / f'{comp_name}_{j}.png'
                    else:
                        img_path = base_dir / 'SEN2SR' / f'{comp_name}_{j}.png'
                    img = plt.imread(img_path)
                    axs[i].imshow(img)
                    axs[i].set_title(method)
                    axs[i].axis('off')
                plt.tight_layout()
                plt.savefig(base_dir / 'comparaisons' / f'comparaison_{comp_name}_{j}.png', dpi=150)
                plt.close()
        print(f"\nTraitement terminé pour {zone['nom']}!")
        print(f"Total d'images traitées: {n_images}")
        print(f"Résultats disponibles dans: {base_dir}")

if __name__ == "__main__":
    main()
