from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .models import Transcription



def transcribe_audio(request):
    if request.method == 'POST' and 'audio' in request.FILES:
        audio_file = request.FILES['audio']
        # Perform speech-to-text transcription here (using external libraries or services like Google Cloud Speech-to-Text).
        # Save the transcribed text to the database.
        transcribed_text = ""
        Transcription.objects.create(text=transcribed_text)
        return JsonResponse({'success': True, 'text': transcribed_text})
    return JsonResponse({'success': False, 'error': 'Invalid request'})


