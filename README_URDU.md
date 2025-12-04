# ChatterBox Voice Cloning TTS Service

یہ Python-based Flask service ہے جو ChatterBox voice cloning model استعمال کرتی ہے۔

## Installation / تنصیب

### Step 1: Python Environment بنائیں

```bash
# Python 3.9+ درکار ہے
python --version

# Virtual environment بنائیں
python -m venv venv

# Activate کریں
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
.\venv\Scripts\activate.bat
```

### Step 2: Dependencies Install کریں

```bash
pip install -r requirements.txt
```

**Note:** PyTorch کی installation system کے مطابق مختلف ہو سکتی ہے:

```bash
# CPU only (کم resources):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU support (NVIDIA CUDA):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Service شروع کریں

```bash
python app.py
```

Service `http://localhost:5001` پر چلے گی۔

## API Endpoints

### 1. Health Check
```
GET /health
```

### 2. Text-to-Speech Synthesis
```
POST /api/tts/synthesize
Content-Type: application/json

{
  "text": "ٹکٹ نمبر 101 براہ کرم کاؤنٹر 5 پر تشریف لائیں",
  "language": "ur",
  "speed": 1.0,
  "pitch": 1.0,
  "voice_sample": "optional_voice_id"
}
```

### 3. Upload Voice Sample (Voice Cloning کے لیے)
```
POST /api/tts/upload-voice
Content-Type: multipart/form-data

voice: <audio file>
```

### 4. List Available Voices
```
GET /api/tts/voices
```

## Usage in Queue Management System

### Node.js Backend Integration

`backend/routes/voices.js` میں:

```javascript
import express from 'express';
import axios from 'axios';

const router = express.Router();
const PYTHON_TTS_URL = 'http://localhost:5001';

// Generate speech
router.post('/generate', async (req, res) => {
  try {
    const { text, language, speed, pitch } = req.body;
    
    const response = await axios.post(`${PYTHON_TTS_URL}/api/tts/synthesize`, {
      text,
      language,
      speed,
      pitch
    });
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
```

## System Requirements

- **Python:** 3.9 یا اس سے اوپر
- **RAM:** کم از کم 4GB (8GB recommended)
- **Storage:** ~5GB (models کے لیے)
- **GPU:** Optional لیکن بہتر performance کے لیے recommended

## Troubleshooting

### Issue: Models download نہیں ہو رہیں
**Solution:** Internet connection check کریں اور Hugging Face accessible ہے

### Issue: Out of Memory error
**Solution:** Smaller batch size استعمال کریں یا CPU mode میں چلائیں

### Issue: Slow inference
**Solution:** GPU استعمال کریں یا model quantization کریں

## Production Deployment

Production میں deploy کرنے کے لیے:

1. **Gunicorn استعمال کریں:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

2. **Docker استعمال کریں** (آگے instructions آ رہی ہیں)

3. **Cloud Deploy کریں** (AWS, Azure, Google Cloud)

## Notes

- یہ service background میں چلتی رہے گی
- Node.js backend اس کو HTTP requests کے ذریعے استعمال کرے گی
- Audio files `generated_audio/` folder میں save ہوں گی
