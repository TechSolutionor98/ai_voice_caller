# TTS Implementation Install Guide
# TTS Implementation انسٹال گائیڈ

## Step 1: Install New Dependencies

```powershell
# Activate virtual environment (agar already active nahi hai)
cd python-tts-service
.\venv\Scripts\Activate.ps1

# Install new TTS packages
pip install pyttsx3==2.90
pip install gTTS==2.4.0
pip install pydub==0.25.1
```

## Step 2: Install FFmpeg (Required for pydub)

### Option A: Using Chocolatey (Recommended)

```powershell
# Administrator PowerShell mein:
choco install ffmpeg
```

### Option B: Manual Download

1. Download FFmpeg: https://www.gyan.dev/ffmpeg/builds/
2. Extract to: `C:\ffmpeg`
3. Add to PATH:
   - System Properties > Environment Variables
   - Edit PATH variable
   - Add: `C:\ffmpeg\bin`

### Option C: Skip FFmpeg (Basic functionality)

Agar FFmpeg install nahi karna chahte, toh code automatically fallback karega lekin audio quality kam hogi.

## Step 3: Restart Python Service

```powershell
# Service band karein (Ctrl+C)
# Phir dobara start karein:
python app.py
```

## Step 4: Test TTS

Configuration page pe test karein:
1. Text enter karein
2. "Test Voice" daba dein
3. Audio play hona chahiye!

---

## TTS Engines Available:

1. **gTTS (Google Text-to-Speech)**
   - ✅ Best quality
   - ✅ Multiple languages
   - ⚠️ Requires internet

2. **pyttsx3 (Offline TTS)**
   - ✅ Works offline
   - ✅ Fast
   - ⚠️ Limited voices

3. **Placeholder Audio**
   - ✅ Always works
   - ⚠️ Just a beep tone

System automatically tries engines in order until one works!

---

## Troubleshooting:

### Error: "No module named 'pyttsx3'"
```powershell
pip install pyttsx3 gTTS pydub
```

### Error: "FFmpeg not found"
Install FFmpeg (see Option A or B above)

### gTTS not working:
Check internet connection

---

**Ready!** Ab actual TTS kaam karega! 🎙️✨
