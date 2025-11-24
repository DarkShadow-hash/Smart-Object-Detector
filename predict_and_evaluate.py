import os
from ultralytics import YOLO

# --- Chemins d'accès importants ---
# Assurez-vous d'adapter ces chemins à votre structure locale
PATH_TO_CUSTOM_MODEL = './runs/detect/train2/weights/best.pt' 
PATH_TO_TEST_IMAGES = './images_to_test/'
PATH_TO_BASELINE_MODEL = 'yolov8n.pt'  # Le modèle de base téléchargé automatiquement par ultralytics

# --- Noms des classes (IMPORTANT : doit correspondre au data.yaml) ---
# Roboflow les a nommés : 0: 'bottle of water', 1: 'headphones', 2: 'pencil case'
NAMES = {0: 'Mont Roucous', 1: 'Casque', 2: 'Trousse'} 


def run_prediction(model_path, source_dir, project_name):
    """
    Charge et exécute le modèle de détection sur l'ensemble d'images spécifié.
    
    Args:
        model_path (str): Chemin d'accès aux poids du modèle (ex: 'yolov8n.pt' ou 'best.pt').
        source_dir (str): Chemin d'accès aux images à tester.
        project_name (str): Nom du sous-dossier où stocker les résultats dans ./runs/detect/
    """
    # 1. Charger le modèle
    print(f"\n--- Chargement du modèle : {project_name} ---")
    model = YOLO(model_path)
    
    # 2. Exécuter la prédiction
    # On ajuste le seuil de confiance ('conf') pour être plus rigoureux dans la détection.
    # L'output est sauvegardé dans /runs/detect/
    results = model.predict(
        source=source_dir, 
        conf=0.25,      # Seuil de confiance (ajustez ce paramètre pour l'évaluation)
        save=True,      # Sauvegarde les images avec les boîtes
        name=project_name # Crée un sous-dossier dans runs/detect/
    )
    
    # 3. Afficher les métriques de base
    # (Facultatif, mais ajoute de la rigueur)
    metrics = model.val(data='./labelling images.v2i.yolov8/data.yaml', split='test') # Évaluation sur le VRAI ensemble de test
    print(f"Metrics {project_name} on test set (mAP50): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    return results


if __name__ == '__main__':
    # 1. Tester le modèle Baseline (pour montrer que c'est "pas ouf")
    run_prediction(
        model_path=PATH_TO_BASELINE_MODEL,
        source_dir=PATH_TO_TEST_IMAGES,
        project_name='baseline_yolo' 
    )

    # 2. Tester votre modèle Custom (pour montrer votre contribution)
    run_prediction(
        model_path=PATH_TO_CUSTOM_MODEL,
        source_dir=PATH_TO_TEST_IMAGES,
        project_name='custom_trained'
    )
    
    # Note: Les images avec détections seront sauvegardées dans ./runs/detect/
