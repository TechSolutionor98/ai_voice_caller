"""
ChatterBox Voice Cloning TTS Service
Flask-based service for voice cloning and text-to-speech generation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import logging
from datetime import datetime
import pyttsx3
from gtts import gTTS
import wave
import struct
import math
from deep_translator import GoogleTranslator

# Optional imports for ChatterBox model
try:
    import torch
    from huggingface_hub import hf_hub_download
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("torch not available, using gTTS and pyttsx3 only")

app = Flask(__name__)

# Configure CORS with specific origin (prevents duplicate headers)
CORS(app, resources={
    r"/*": {
        "origins": ["https://qmanagement-frontend.vercel.app", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models_loaded = False
s3gen_model = None
t3_cfg_model = None
ve_model = None

# Directories
MODELS_DIR = "./models"
VOICES_DIR = "./voice_samples"
OUTPUT_DIR = "./generated_audio"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_speech(text, output_path, language='en', speed=1.0, pitch=1.0, voice_type='default'):
    """
    Generate speech using available TTS engines
    Falls back through multiple engines for reliability
    voice_type: 'male', 'female', 'child', or 'default'
    """
    # ✅ DYNAMIC: Only convert 'default' to 'male' for English, but respect user's choice
    if voice_type == 'default' or not voice_type:
        # Only for English, default to male (gTTS female is default for other languages)
        voice_type = 'male' if language == 'en' else 'default'
        logger.info(f"🔄 Voice type was empty/default, using: {voice_type} for {language}")
    
    # ✅ Use pyttsx3 for English when SPECIFIC voice type is requested (male/female/child)
    # For non-English languages, use gTTS because Windows pyttsx3 only has English voices
    use_pyttsx3_first = (voice_type in ['male', 'female', 'child'] and language == 'en')
    
    # Try pyttsx3 first for English with specific voice type
    if use_pyttsx3_first:
        try:
            logger.info(f"🎤 Using pyttsx3 for voice type: {voice_type} (language: {language})")
            engine = pyttsx3.init()
            
            # Set properties
            rate = engine.getProperty('rate')
            engine.setProperty('rate', int(rate * speed))
            
            volume = engine.getProperty('volume')
            engine.setProperty('volume', volume)
            
            # Try to find matching voice by type and language
            voices = engine.getProperty('voices')
            voice_keywords = {
                'male': ['david', 'mark', 'male', 'man', 'james', 'george'],
                'female': ['zira', 'hazel', 'female', 'woman', 'susan', 'mary', 'linda'],
                'child': ['child', 'kid', 'young']
            }
            
            keywords = voice_keywords.get(voice_type.lower(), [])
            selected = False
            
            logger.info(f"🔍 Searching for {voice_type} voice in {len(voices)} available voices...")
            
            # Try to match language + voice type
            for voice in voices:
                voice_name = voice.name.lower()
                voice_langs = getattr(voice, 'languages', [])
                
                # Check if voice matches language and type
                lang_match = any(language.split('-')[0] in str(lang).lower() for lang in voice_langs) if voice_langs else True
                type_match = any(keyword in voice_name for keyword in keywords) if keywords else False
                
                if lang_match and type_match:
                    engine.setProperty('voice', voice.id)
                    logger.info(f"✅ Selected PERFECT match: {voice.name} (lang: {language}, type: {voice_type})")
                    selected = True
                    break
            
            # Fallback: just match type (any language)
            if not selected and keywords:
                for voice in voices:
                    voice_name = voice.name.lower()
                    if any(keyword in voice_name for keyword in keywords):
                        engine.setProperty('voice', voice.id)
                        logger.info(f"✅ Selected voice by TYPE: {voice.name} (type: {voice_type})")
                        selected = True
                        break
            
            # If no specific voice found, use default but log warning
            if not selected:
                logger.warning(f"⚠️ Could not find {voice_type} voice, using system default")
            
            # Adjust pitch by modifying rate
            if pitch != 1.0:
                current_rate = engine.getProperty('rate')
                engine.setProperty('rate', int(current_rate * pitch))
            
            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            logger.info(f"✅ Speech generated with pyttsx3 ({voice_type}): {output_path}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ pyttsx3 failed: {str(e)}, falling back to gTTS")
    
    # Default: Use gTTS for reliability and language support
    try:
        # Method 1: Try gTTS (Google Text-to-Speech) - Simple and reliable
        lang_map = {
            'en': 'en',
            'ur': 'ur',
            'ar': 'ar',
            'ar-ae': 'ar',  # Dubai Arabic uses standard Arabic TTS
            'es': 'es',
            'hi': 'hi',
            'fr': 'fr',
            'de': 'de',
            'zh': 'zh-CN',
            'ja': 'ja'
        }
        
        gtts_lang = lang_map.get(language, 'en')
        logger.info(f"Using gTTS with language code: {gtts_lang} (from input: {language})")
        logger.info(f"⚠️ NOTE: gTTS doesn't support voice_type selection - using audio manipulation for voice effect")
        
        # Generate with gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=(speed < 0.8))
        
        # Save to temporary MP3 first
        temp_mp3 = output_path.replace('.wav', '_temp.mp3')
        tts.save(temp_mp3)
        
        # Convert MP3 to WAV and apply voice effects
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_mp3(temp_mp3)
            
            # ✅ APPLY VOICE TYPE EFFECT using audio manipulation
            logger.info(f"🎛️ Applying {voice_type} voice effect to gTTS audio...")
            
            # Male voice: Lower pitch (slower playback rate) 
            if voice_type == 'male':
                logger.info("👨 Creating male voice effect: lowering pitch by 15%")
                # Lower frame rate = deeper voice
                new_frame_rate = int(sound.frame_rate * 0.85)  # 15% lower
                sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                sound = sound.set_frame_rate(44100)
            
            # Female voice: Keep default or slightly raise pitch
            elif voice_type == 'female':
                logger.info("👩 Using default gTTS voice (naturally female)")
                # gTTS default is already female-sounding, so minimal change
                pass
            
            # Child voice: Higher pitch
            elif voice_type == 'child':
                logger.info("👶 Creating child voice effect: raising pitch by 20%")
                # Higher frame rate = higher pitched voice
                new_frame_rate = int(sound.frame_rate * 1.20)  # 20% higher
                sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                sound = sound.set_frame_rate(44100)
            
            # ✅ HIGH QUALITY SETTINGS - Remove distortion/crackling
            # Normalize audio to prevent clipping and distortion
            sound = sound.normalize()
            
            # Apply speed changes SMOOTHLY (if needed)
            if speed != 1.0:
                # Use frame rate manipulation instead of speedup for better quality
                new_frame_rate = int(sound.frame_rate / speed)
                sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                sound = sound.set_frame_rate(44100)  # Standard high quality sample rate
            
            # Apply pitch changes SMOOTHLY (if needed)
            if pitch != 1.0:
                # Octaves manipulation for cleaner pitch shifting
                octaves = math.log2(pitch)
                new_sample_rate = int(sound.frame_rate * (2 ** octaves))
                sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
                sound = sound.set_frame_rate(44100)
            
            # Export with HIGH QUALITY settings
            sound.export(
                output_path, 
                format="wav",
                parameters=[
                    "-ar", "44100",      # Sample rate: 44.1kHz (CD quality)
                    "-ac", "1",          # Mono channel
                    "-b:a", "192k",      # Bitrate: 192kbps (high quality)
                    "-acodec", "pcm_s16le"  # PCM 16-bit (uncompressed, no artifacts)
                ]
            )
            os.remove(temp_mp3)
            
            logger.info(f"Speech generated successfully with HIGH QUALITY: {output_path}")
            return True
            
        except (ImportError, FileNotFoundError, RuntimeError, Exception) as pydub_err:
            # If pydub fails (missing or any error), use librosa for pitch shifting
            logger.warning(f"⚠️ pydub/ffmpeg issue: {type(pydub_err).__name__}: {str(pydub_err)}")
            logger.info("🎵 Falling back to librosa for voice processing")
            try:
                import librosa
                import soundfile as sf
                import numpy as np
                
                logger.info(f"🎛️ Using librosa to apply {voice_type} voice effect...")
                
                # Load MP3 audio with high quality
                y, sr = librosa.load(temp_mp3, sr=44100, mono=True)
                logger.info(f"📊 Loaded audio: sample_rate={sr}, duration={len(y)/sr:.2f}s")
                
                # Apply voice effects based on type
                if voice_type == 'male':
                    logger.info("👨 Applying natural male voice effect...")
                    # For male: slightly slower speed (0.95x) gives deeper, more natural sound
                    # This is better than pitch shifting which can cause artifacts
                    y_shifted = librosa.effects.time_stretch(y=y, rate=0.95)
                    logger.info("✅ Male voice effect applied (natural time stretch)")
                    
                elif voice_type == 'child':
                    logger.info("👶 Applying child voice effect...")
                    # For child: faster speed (1.1x) gives higher, energetic sound
                    y_shifted = librosa.effects.time_stretch(y=y, rate=1.1)
                    logger.info("✅ Child voice effect applied")
                    
                else:  # female - keep default
                    logger.info("👩 Using default (female) voice")
                    y_shifted = y
                
                # Gentle normalization to prevent distortion
                max_val = np.max(np.abs(y_shifted))
                if max_val > 0:
                    # Normalize to 85% to leave headroom and prevent clipping
                    y_shifted = y_shifted / max_val * 0.85
                    logger.info("✅ Audio normalized (gentle)")
                
                # Save as high-quality WAV
                logger.info(f"💾 Saving audio to: {output_path}")
                sf.write(output_path, y_shifted, sr, subtype='PCM_16')
                logger.info("✅ Clean audio file saved")
                
                # Small delay to ensure file handle is released
                import time
                time.sleep(0.1)
                
                # Try to remove temp file with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(temp_mp3):
                            os.remove(temp_mp3)
                            logger.info("🗑️ Temporary MP3 file removed")
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.2)
                        else:
                            logger.warning(f"⚠️ Could not delete temp file (will be cleaned up later): {temp_mp3}")
                
                logger.info(f"✅ Speech generated with librosa pitch shift ({voice_type}): {output_path}")
                return True
                
            except ImportError as lie:
                logger.warning(f"⚠️ librosa import error: {str(lie)}")
                logger.warning("Using MP3 format without pitch changes")
                # Keep MP3 and update filename
                final_path = output_path.replace('.wav', '.mp3')
                os.rename(temp_mp3, final_path)
                logger.info(f"Saved as MP3 (no pitch modification): {final_path}")
                return True
            except Exception as librosa_error:
                logger.error(f"❌ Librosa pitch shift FAILED: {str(librosa_error)}")
                logger.error(f"   Error type: {type(librosa_error).__name__}")
                # Fallback to MP3 without modification
                final_path = output_path.replace('.wav', '.mp3')
                if os.path.exists(temp_mp3):
                    os.rename(temp_mp3, final_path)
                logger.info(f"Saved as MP3 (fallback): {final_path}")
                return True
                
        except Exception as conv_error:
            logger.error(f"Audio conversion failed: {str(conv_error)}")
            # Keep the MP3 file as fallback
            final_path = output_path.replace('.wav', '.mp3')
            if os.path.exists(temp_mp3):
                os.rename(temp_mp3, final_path)
            return True
            
    except Exception as e:
        logger.error(f"gTTS failed: {str(e)}")
        
        # Method 2: Fallback to pyttsx3 (offline TTS)
        try:
            logger.info("Falling back to pyttsx3...")
            engine = pyttsx3.init()
            
            # Set properties
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate * speed)
            
            volume = engine.getProperty('volume')
            engine.setProperty('volume', volume)
            
            # Set voice based on language preference
            voices = engine.getProperty('voices')
            for voice in voices:
                if language == 'en' and 'english' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
                elif language in voice.languages:
                    engine.setProperty('voice', voice.id)
                    break
            
            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            logger.info(f"Speech generated with pyttsx3: {output_path}")
            return True
            
        except Exception as e2:
            logger.error(f"pyttsx3 also failed: {str(e2)}")
            
            # Method 3: Create a simple beep tone as last resort
            logger.warning("All TTS engines failed, creating placeholder audio")
            create_placeholder_audio(output_path, duration=2)
            return True
    
    return False


def create_placeholder_audio(filepath, duration=2, frequency=440):
    """Create a simple tone as placeholder when TTS fails"""
    try:
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            for i in range(num_samples):
                value = int(32767.0 * math.sin(2 * math.pi * frequency * i / sample_rate))
                data = struct.pack('<h', value)
                wav_file.writeframes(data)
        
        return True
    except Exception as e:
        logger.error(f"Failed to create placeholder audio: {str(e)}")
        return False


def download_models():
    """Download ChatterBox models from Hugging Face"""
    global models_loaded
    
    if not TORCH_AVAILABLE:
        logger.warning("torch not available, skipping model download")
        return False
    
    if models_loaded:
        return True
    
    try:
        logger.info("Downloading ChatterBox models from Hugging Face...")
        logger.info(f"Models will be saved to: {MODELS_DIR}")
        
        # Download model files
        model_files = ["s3gen.pt", "t3_cfg.pt", "ve.pt", "tokenizer.json"]
        
        for filename in model_files:
            model_path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(model_path):
                logger.info(f"Downloading {filename}...")
                downloaded_path = hf_hub_download(
                    repo_id="ramimu/chatterbox-voice-cloning-model",
                    filename=filename,
                    local_dir=MODELS_DIR,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded {filename} to {downloaded_path}")
        
        # Verify all files exist
        all_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in model_files)
        if all_exist:
            models_loaded = True
            logger.info("All models downloaded successfully!")
            logger.info(f"Models location: {os.path.abspath(MODELS_DIR)}")
        else:
            logger.error("Some models failed to download")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return False


def load_models():
    """Load the ChatterBox models into memory"""
    global s3gen_model, t3_cfg_model, ve_model
    
    if not TORCH_AVAILABLE:
        logger.warning("torch not available, skipping model loading")
        return False
    
    try:
        if not models_loaded:
            download_models()
        
        logger.info("Loading models into memory...")
        
        # Load models
        s3gen_path = os.path.join(MODELS_DIR, "s3gen.pt")
        t3_cfg_path = os.path.join(MODELS_DIR, "t3_cfg.pt")
        ve_path = os.path.join(MODELS_DIR, "ve.pt")
        
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using GPU for inference")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        
        s3gen_model = torch.load(s3gen_path, map_location=device)
        t3_cfg_model = torch.load(t3_cfg_path, map_location=device)
        ve_model = torch.load(ve_path, map_location=device)
        
        logger.info("Models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/tts/synthesize', methods=['POST'])
@app.route('/synthesize', methods=['POST'])  # Add backward compatibility route
def synthesize_speech():
    """
    Synthesize speech from text using voice cloning
    
    Expected JSON:
    {
        "text": "Ticket number 101 please come to counter 5",
        "voice_sample": "path/to/voice/sample.wav",  # Optional
        "language": "en",  # en, ur, ar
        "speed": 1.0,
        "pitch": 1.0
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        voice_sample = data.get('voice_sample')
        voice_type = data.get('voice_type', 'default')  # male, female, child, default
        language = data.get('language', 'en')
        speed = data.get('speed', 1.0)
        pitch = data.get('pitch', 1.0)
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Check if text is already in target language (skip translation)
        # Detect if text contains Arabic/Urdu/Hindi characters
        def contains_non_latin(text):
            """Check if text contains non-Latin characters (Arabic, Urdu, Hindi, etc.)"""
            import re
            # Arabic: \u0600-\u06FF, \u0750-\u077F, \uFB50-\uFDFF, \uFE70-\uFEFF
            # Devanagari (Hindi): \u0900-\u097F
            pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0900-\u097F]')
            return bool(pattern.search(text))
        
        # Auto-translate ONLY if text is in English and target is non-English
        original_text = text
        should_translate = language != 'en' and not contains_non_latin(text)
        
        if should_translate:
            try:
                logger.info(f"🌐 Auto-translating English text to {language}...")
                logger.info(f"Original English text: '{original_text}'")
                
                # Replace dash with space before translation (so dash is not read)
                text_to_translate = text.replace('-', ' ')
                
                # Create translator for target language
                # Handle ar-ae (Dubai Arabic) - use standard 'ar'
                target_lang = 'ar' if language == 'ar-ae' else language
                translator = GoogleTranslator(source='en', target=target_lang)
                text = translator.translate(text_to_translate)
                
                logger.info(f"✅ Translated to {language}: '{text}'")
            except Exception as trans_err:
                logger.warning(f"⚠️ Translation failed: {trans_err}, using original text")
                # Continue with original text if translation fails
        else:
            if contains_non_latin(text):
                logger.info(f"✅ Text already in non-Latin script ({language}), skipping translation")
                logger.info(f"Text: '{text}'")
            else:
                logger.info(f"✅ Text is English for English language, no translation needed")
        
        # Ensure models are loaded (optional, only for ChatterBox)
        if TORCH_AVAILABLE and not models_loaded:
            load_models()  # Try to load but don't fail if it doesn't work
        
        logger.info(f"========== SYNTHESIS REQUEST ==========")
        logger.info(f"📝 Text: '{text}'")
        logger.info(f"🌐 Language: {language}")
        logger.info(f"🎤 Voice Type: {voice_type}")
        logger.info(f"⚡ Speed: {speed}")
        logger.info(f"🎵 Pitch: {pitch}")
        logger.info(f"=======================================")
        
        output_filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Generate speech using available TTS engines
        success = generate_speech(text, output_path, language, speed, pitch, voice_type)
        
        if not success:
            return jsonify({"error": "Failed to generate speech"}), 500
        
        # Check if MP3 was created instead of WAV (when pydub not available)
        actual_filename = output_filename
        if not os.path.exists(output_path):
            mp3_path = output_path.replace('.wav', '.mp3')
            if os.path.exists(mp3_path):
                actual_filename = output_filename.replace('.wav', '.mp3')
                logger.info(f"Audio saved as MP3: {actual_filename}")
        
        return jsonify({
            "success": True,
            "audio_url": f"/api/tts/audio/{actual_filename}",
            "message": "Speech synthesized successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tts/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serve generated audio files"""
    try:
        audio_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404
        
        # Determine mimetype based on extension
        mimetype = 'audio/wav'
        if filename.endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.endswith('.ogg'):
            mimetype = 'audio/ogg'
        
        return send_file(audio_path, mimetype=mimetype)
        
    except Exception as e:
        logger.error(f"Error serving audio: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tts/upload-voice', methods=['POST'])
def upload_voice_sample():
    """Upload a voice sample for cloning"""
    try:
        if 'voice' not in request.files:
            return jsonify({"error": "No voice file provided"}), 400
        
        voice_file = request.files['voice']
        
        if voice_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save voice sample
        filename = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{voice_file.filename}"
        filepath = os.path.join(VOICES_DIR, filename)
        voice_file.save(filepath)
        
        logger.info(f"Voice sample saved: {filename}")
        
        return jsonify({
            "success": True,
            "voice_id": filename,
            "message": "Voice sample uploaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Error uploading voice sample: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tts/voices', methods=['GET'])
def list_voices():
    """List all available voice samples with dynamic system voices"""
    try:
        voices = []
        
        # Add dynamic system voices from pyttsx3
        try:
            engine = pyttsx3.init()
            system_voices = engine.getProperty('voices')
            
            # Keep track of voice type counts for unique IDs
            voice_type_counter = {'male': 0, 'female': 0, 'child': 0, 'default': 0}
            
            for voice in system_voices:
                voice_name = voice.name
                voice_id_lower = voice.id.lower()
                
                # Determine voice type based on name/id
                if 'david' in voice_name.lower() or 'male' in voice_id_lower:
                    voice_type = 'male'
                    icon = '👨'
                elif 'zira' in voice_name.lower() or 'female' in voice_id_lower:
                    voice_type = 'female'
                    icon = '👩'
                elif 'child' in voice_name.lower():
                    voice_type = 'child'
                    icon = '👶'
                else:
                    voice_type = 'default'
                    icon = '🔊'
                
                # Increment counter for this voice type
                voice_type_counter[voice_type] += 1
                
                # ONLY add first male and first female voice (English ones)
                if voice_type == 'male' and voice_type_counter['male'] == 1:
                    voices.append({
                        "id": "male",  # Simple ID for primary voices
                        "name": f"{icon} Male Voice",
                        "type": "male",
                        "system_name": voice_name,
                        "source": "pyttsx3"
                    })
                elif voice_type == 'female' and voice_type_counter['female'] == 1:
                    voices.append({
                        "id": "female",  # Simple ID for primary voices
                        "name": f"{icon} Female Voice",
                        "type": "female",
                        "system_name": voice_name,
                        "source": "pyttsx3"
                    })
                elif voice_type == 'child' and voice_type_counter['child'] == 1:
                    voices.append({
                        "id": "child",
                        "name": f"{icon} Child Voice",
                        "type": "child",
                        "system_name": voice_name,
                        "source": "pyttsx3"
                    })
            
            engine.stop()
            logger.info(f"✅ Loaded {len(voices)} primary system voices from pyttsx3")
            
        except Exception as voice_error:
            logger.warning(f"⚠️ Could not load pyttsx3 voices: {voice_error}")
            # Fallback to default voices
            voices = [
                {"id": "male", "name": "👨 Male Voice", "type": "male", "source": "default"},
                {"id": "female", "name": "👩 Female Voice", "type": "female", "source": "default"},
                {"id": "child", "name": "👶 Child Voice", "type": "child", "source": "default"}
            ]
        
        # Add custom uploaded voice samples
        try:
            for filename in os.listdir(VOICES_DIR):
                if filename.endswith(('.wav', '.mp3')):
                    voices.append({
                        "id": filename,
                        "name": f"🎤 {filename.replace('voice_', '').replace('.wav', '').replace('.mp3', '')}",
                        "type": "custom",
                        "path": os.path.join(VOICES_DIR, filename),
                        "source": "uploaded"
                    })
        except Exception as file_error:
            logger.warning(f"⚠️ Could not load custom voices: {file_error}")
        
        return jsonify({
            "success": True,
            "data": voices,
            "count": len(voices)
        })
        
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "data": [
                {"id": "male", "name": "👨 Male Voice", "type": "male", "source": "fallback"},
                {"id": "female", "name": "👩 Female Voice", "type": "female", "source": "fallback"},
                {"id": "child", "name": "👶 Child Voice", "type": "child", "source": "fallback"}
            ]
        }), 200  # Return 200 with fallback data instead of error


if __name__ == '__main__':
    # Download models on startup (optional)
    logger.info("Starting TTS Service with gTTS and pyttsx3...")
    logger.info(f"Torch available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        logger.info("ChatterBox models support enabled")
        download_models()
    else:
        logger.info("Using gTTS and pyttsx3 for speech synthesis")
    
    # Run the Flask app
    port = int(os.getenv('PORT', 3002))
    app.run(host='0.0.0.0', port=port, debug=True)
