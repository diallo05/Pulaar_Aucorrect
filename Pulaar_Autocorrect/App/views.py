import os
import sys
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Add the App directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))
from spell_corrector import load_model, correct_text

# Charger le modèle au démarrage
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pulaar_spell_model.pkl")
probabilities = load_model(MODEL_PATH)


def index(request):
    """Page d'accueil avec le formulaire de correction."""
    return render(request, 'index.html')


@csrf_exempt
def correct(request):
    """API endpoint pour corriger un texte."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text:
                return JsonResponse({'error': 'Aucun texte fourni'}, status=400)
            
            # Corriger le texte
            corrected_text = correct_text(text, probabilities)
            
            return JsonResponse({
                'original_text': text,
                'corrected_text': corrected_text,
                'success': True
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)


def health(request):
    """Endpoint de santé pour vérifier que l'application fonctionne."""
    return JsonResponse({
        'status': 'ok',
        'model_loaded': len(probabilities) > 0,
        'vocabulary_size': len(probabilities)
    })
