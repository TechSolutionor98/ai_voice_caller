# Fix Python TTS Service Warnings

## ⚠️ Current Issues

Your TTS service is working BUT has these warnings:
1. ❌ `eSpeak or eSpeak-ng not installed` - pyttsx3 fails
2. ❌ `ffmpeg/ffprobe not found` - Audio manipulation fails  
3. ❌ `librosa not installed` - Pitch/speed changes don't work

## ✅ Solution

### Step 1: Install Missing Python Libraries

```bash
cd python-tts-service
pip install librosa soundfile
```

### Step 2: Install ffmpeg (Required for audio manipulation)

#### **Windows:**
```powershell
# Using Chocolatey (recommended)
choco install ffmpeg

# OR Download manually from:
# https://www.gyan.dev/ffmpeg/builds/
# Extract and add to PATH
```

#### **Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg espeak
```

#### **macOS:**
```bash
brew install ffmpeg espeak
```

### Step 3: Install eSpeak (Optional - for pyttsx3)

#### **Windows:**
Download from: http://espeak.sourceforge.net/download.html

#### **Linux:**
```bash
sudo apt-get install espeak
```

#### **macOS:**
```bash
brew install espeak
```

### Step 4: Restart Python TTS Service

```bash
cd python-tts-service
python app.py
```

---

## 🔍 Why These Are Important:

- **ffmpeg**: Required for changing pitch/speed and converting audio formats
- **librosa**: Python library for advanced audio processing
- **eSpeak**: Offline text-to-speech engine (fallback when internet is down)

Without these, the service works but uses Google TTS without voice modifications.

---

## 🎯 Expected Result After Fix:

```
INFO:__main__:🎤 Using pyttsx3 for voice type: female (language: en)
INFO:__main__:✅ Pitch modified successfully using pydub
INFO:__main__:Audio saved as WAV: speech_xxxxx.wav
```

No more warnings! 🎉
