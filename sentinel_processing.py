import ee
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from skimage import filters
import os

# === 1. CONFIGURATION ===
def main():
    # Authentification Earth Engine
    try:
        ee.Initialize(project='tidy-bindery-461215-i7')
    except:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize(project='tidy-bindery-461215-i7')
    
    # Définition des zones forestières
    zones = {
        "foret_moreno": {
            "nom": "Forêt de la Moreno",
            "roi": ee.Geometry.BBox(2.94, 45.74, 2.99, 45.77),
        },
        "combe_valtin": {
            "nom": "Combe Valtin",
            "roi": ee.Geometry.BBox(6.98, 48.10, 7.03, 48.13),
        },
        "foret_preny": {
            "nom": "Forêt de Preny",
            "roi": ee.Geometry.BBox(6.05, 48.73, 6.10, 48.76),
        }
    }
    
    # Paramètres communs
    DATE_DEBUT = '2024-05-01'
    DATE_FIN = '2024-05-31'
    MAX_NUAGES = 10
    BANDES = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']
    
    # === 2. TRAITEMENT PAR ZONE ===
    for zone_id, zone in zones.items():
        print(f"\n=== Traitement de {zone['nom']} ===")
        base_dir = Path('forets_comparaison') / zone_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dossiers de sortie
        (base_dir / 'brut').mkdir(exist_ok=True)
        (base_dir / 'DSen2').mkdir(exist_ok=True)
        (base_dir / 'CESBIO_SR').mkdir(exist_ok=True)
        (base_dir / 'SEN2SR').mkdir(exist_ok=True)
        (base_dir / 'comparaisons').mkdir(exist_ok=True)
        
        # Téléchargement de l'image
        print("Recherche d'image Sentinel-2...")
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(zone['roi'])
                      .filterDate(DATE_DEBUT, DATE_FIN)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUAGES)))
        
        if collection.size().getInfo() == 0:
            print("Aucune image trouvée - Élargissement des critères")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(zone['roi'])
                          .filterDate('2024-01-01', '2024-06-30')
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
        
        image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        image_id = image.get('system:id').getInfo()
        
        url = image.getDownloadURL({
            'bands': BANDES,
            'region': zone['roi'],
            'scale': 10,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:32631'
        })
        
        # Téléchargement
        raw_path = base_dir / 'brut' / f"raw_{image_id.replace('/', '_')}.tif"
        response = requests.get(url)
        
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        print(f"Image brute téléchargée: {raw_path}")
        
        # === 3. COMPOSITIONS DE BASE ===
        with rasterio.open(raw_path) as src:
            data = src.read()
            band_names = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']
            bands = {band: data[i] for i, band in enumerate(band_names)}
            
            # Fonction de normalisation
            def normalize(band):
                return (band - np.min(band)) / (np.max(band) - np.min(band) + 1e-10)
            
            # Sauvegarde des compositions brutes
            compositions = {
                'rgb': np.stack([normalize(bands['B4']), normalize(bands['B3']), normalize(bands['B2'])], axis=-1),
                'ndvi': (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'] + 1e-10),
                'false_color': np.stack([normalize(bands['B5']), normalize(bands['B8']), normalize(bands['B11'])], axis=-1)
            }
            
            for comp_name, comp_data in compositions.items():
                if comp_name == 'ndvi':
                    plt.imsave(base_dir / 'brut' / f'{comp_name}.png', comp_data, cmap='RdYlGn', vmin=-1, vmax=1)
                else:
                    plt.imsave(base_dir / 'brut' / f'{comp_name}.png', comp_data)
        
        # === 4. SIMULATION DES MÉTHODES (BANDE PAR BANDE) ===
        def simulate_band(band_data, factor, sharpness=True):
            """Simule l'amélioration de résolution pour une bande unique"""
            resized = resize(band_data, 
                            (band_data.shape[0] * factor, 
                             band_data.shape[1] * factor), 
                            anti_aliasing=True)
            if sharpness:
                return filters.unsharp_mask(resized, radius=2, amount=1.5)
            return resized
        
        # Paramètres des méthodes
        method_params = {
            'DSen2': {'factor': 1, 'sharpness': True},
            'CESBIO_SR': {'factor': 2, 'sharpness': False},
            'SEN2SR': {'factor': 4, 'sharpness': False}
        }
        
        # Bandes nécessaires pour les compositions
        required_bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
        
        # Traitement pour chaque méthode
        for method, params in method_params.items():
            method_dir = base_dir / method
            print(f"Traitement {method}...")
            
            # Simulation bande par bande
            sim_bands = {}
            for band_name in required_bands:
                sim_bands[band_name] = simulate_band(bands[band_name], 
                                                    params['factor'], 
                                                    params['sharpness'])
            
            # Recalcul des compositions
            comps = {
                'rgb': np.stack([normalize(sim_bands['B4']), 
                                normalize(sim_bands['B3']), 
                                normalize(sim_bands['B2'])], axis=-1),
                'ndvi': (sim_bands['B8'] - sim_bands['B4']) / (sim_bands['B8'] + sim_bands['B4'] + 1e-10),
                'false_color': np.stack([normalize(sim_bands['B5']), 
                                        normalize(sim_bands['B8']), 
                                        normalize(sim_bands['B11'])], axis=-1)
            }
            
            # Sauvegarde
            for comp_name, comp_data in comps.items():
                if comp_name == 'ndvi':
                    plt.imsave(method_dir / f'{comp_name}.png', comp_data, cmap='RdYlGn', vmin=-1, vmax=1)
                else:
                    plt.imsave(method_dir / f'{comp_name}.png', comp_data)
        
        # === 5. COMPARAISONS SYSTÉMATIQUES ===
        methods = ['brut', 'DSen2', 'CESBIO_SR', 'SEN2SR']
        compositions = ['rgb', 'ndvi', 'false_color']
        
        # Génération des grilles comparatives
        for comp_name in compositions:
            fig, axs = plt.subplots(1, 4, figsize=(25, 6))
            fig.suptitle(f"Comparaison {comp_name} - {zone['nom']}", fontsize=16)
            
            for i, method in enumerate(methods):
                if method == 'brut':
                    img_path = base_dir / 'brut' / f'{comp_name}.png'
                else:
                    img_path = base_dir / method / f'{comp_name}.png'
                
                img = plt.imread(img_path)
                axs[i].imshow(img)
                axs[i].set_title(method)
                axs[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(base_dir / 'comparaisons' / f'comparaison_{comp_name}.png', dpi=150)
            plt.close()
        
        print(f"Comparaisons générées pour {zone['nom']}!")

if __name__ == "__main__":
    main()
