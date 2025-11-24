import os
import time
import cv2
from ultralytics import YOLO

# --- Configuration et Chemins (Adaptez les chemins si nécessaire) ---
PATH_TO_CUSTOM_MODEL = './train2_results/weights/best.pt' 
VIDEO_SOURCE = 0 # 0 pour la webcam, ou 'chemin/vers/fichier.mp4'
CONFIDENCE_THRESHOLD = 0.35 # Garder la faible confiance pour compenser l'overfitting
OBJECT_START_TIME = {} # Dictionnaire pour stocker l'ID de l'objet et son temps de début

def track_and_time_objects(model_path, source, conf_thresh):
    """
    Charge le modèle entraîné, exécute la détection sur un flux vidéo 
    (webcam ou fichier), et implémente un chronomètre pour chaque objet 
    détecté, affichant le temps écoulé en secondes.
    
    Args:
        model_path (str): Chemin vers le fichier best.pt.
        source (int/str): 0 pour webcam, ou chemin vers un fichier vidéo.
        conf_thresh (float): Seuil de confiance minimal pour afficher une détection.
    
    Returns:
        None: Affiche la vidéo en temps réel.
    """
    print(f"Chargement du modèle custom : {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erreur de chargement du modèle : {e}")
        return

    # Utiliser OpenCV pour la capture vidéo
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la source vidéo {source}")
        return

    print("Lancement de la détection en temps réel (Appuyez sur 'q' pour quitter)...")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        current_time = time.time()
        
        # 1. Prédiction (avec le modèle custom)
        # On utilise persist=True pour améliorer le suivi entre les trames
        results = model(frame, conf=conf_thresh, verbose=False)
        
        # 2. Traitement des Résultats pour le Chronomètre
        detections = results[0].boxes.data.tolist()
        
        if detections:
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                
                # Créer un ID simple basé sur la position (rudimentaire mais fonctionnel)
                # Nous utilisons l'ID de classe et la position X pour créer une clé unique
                object_id = f"{int(cls)}-{int(x1/50)}" 
                
                # --- Logique du Timer ---
                if object_id not in OBJECT_START_TIME:
                    # L'objet apparaît pour la première fois
                    OBJECT_START_TIME[object_id] = current_time
                
                # Calculer le temps écoulé
                time_elapsed = current_time - OBJECT_START_TIME[object_id]
                time_display = f"({model.names[int(cls)]}) {int(time_elapsed)}s"
                
                # --- Affichage ---
                # Dessiner la boîte de YOLO
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Afficher le texte (Label + Chronomètre)
                cv2.putText(frame, time_display, 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. Affichage de la Trame et Contrôle de Sortie
        cv2.imshow("Detection with Timer (Jour 4)", frame)
        
        # Sortir si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Le test doit être exécuté dans l'environnement virtuel actif
    track_and_time_objects(PATH_TO_CUSTOM_MODEL, VIDEO_SOURCE, CONFIDENCE_THRESHOLD)
