import assemblyai as aai
aai.settings.api_key = "50ebd5adffcf4d4f98bae7a7542ad070"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe(f"Annotations/Voice/371.wav")
print(transcript.text)