"""
Edge-TTS Voice Calling Service
Flask-based service for high-quality multi-language TTS with child boy voice support.
Uses Microsoft Edge TTS as primary engine with gTTS fallback.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import asyncio
import threading
import logging
from datetime import datetime
import edge_tts

# Fallback imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Optional imports for audio processing (fallback pitch shifting)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

import wave
import struct
import math

app = Flask(__name__)

# Configure CORS - Allow all origins for audio serving, specific origins for API endpoints
CORS(app, resources={
    r"/api/tts/audio/*": {
        "origins": "*",  # Allow all origins for audio files
        "methods": ["GET", "HEAD", "OPTIONS"],
        "allow_headers": ["Content-Type", "Range"],
        "expose_headers": ["Content-Length", "Content-Type", "Accept-Ranges"],
        "supports_credentials": False
    },
    r"/*": {
        "origins": [
            "https://qmanagement-frontend.vercel.app",
            "http://localhost:3000",
            "https://qtech.techsolutionor.com",
            "https://qmanagement-frontend-git-main-techsolutionor98.vercel.app",
            "https://qmanagement-frontend-techsolutionor98.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
MODELS_DIR = "./models"
VOICES_DIR = "./voice_samples"
OUTPUT_DIR = "./generated_audio"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# EDGE-TTS VOICE CONFIGURATION
# ============================================================================
# Each language maps to specific voice settings for different voice types.
# For "child" voice: we use a male voice with SSML pitch raised to sound young.
# For "male" voice: standard adult male voice.
# For "female" voice: standard adult female voice.
# ============================================================================

EDGE_TTS_VOICES = {
    # English
    'en': {
        'child': {'voice': 'en-US-AndrewNeural', 'pitch': '+25Hz', 'rate': '+5%'},
        'male': {'voice': 'en-US-GuyNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'en-US-JennyNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'en-US-GuyNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Arabic (UAE / Dubai)
    'ar-ae': {
        'child': {'voice': 'ar-AE-HamdanNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'ar-AE-HamdanNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'ar-AE-FatimaNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'ar-AE-HamdanNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Arabic (Standard)
    'ar': {
        'child': {'voice': 'ar-SA-HamedNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'ar-SA-HamedNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'ar-SA-ZariyahNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'ar-SA-HamedNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Urdu
    'ur': {
        'child': {'voice': 'ur-PK-AsadNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'ur-PK-AsadNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'ur-PK-UzmaNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'ur-PK-AsadNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Hindi
    'hi': {
        'child': {'voice': 'hi-IN-MadhurNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'hi-IN-MadhurNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'hi-IN-SwaraNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'hi-IN-MadhurNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Spanish
    'es': {
        'child': {'voice': 'es-ES-AlvaroNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'es-ES-AlvaroNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'es-ES-ElviraNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'es-ES-AlvaroNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # French
    'fr': {
        'child': {'voice': 'fr-FR-HenriNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'fr-FR-HenriNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'fr-FR-DeniseNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'fr-FR-HenriNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # German
    'de': {
        'child': {'voice': 'de-DE-ConradNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'de-DE-ConradNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'de-DE-KatjaNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'de-DE-ConradNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Chinese
    'zh': {
        'child': {'voice': 'zh-CN-YunxiNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'zh-CN-YunxiNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'zh-CN-XiaoxiaoNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'zh-CN-YunxiNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
    # Japanese
    'ja': {
        'child': {'voice': 'ja-JP-KeitaNeural', 'pitch': '+60Hz', 'rate': '+5%'},
        'male': {'voice': 'ja-JP-KeitaNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'female': {'voice': 'ja-JP-NanamiNeural', 'pitch': '+0Hz', 'rate': '+0%'},
        'default': {'voice': 'ja-JP-KeitaNeural', 'pitch': '+0Hz', 'rate': '+0%'},
    },
}


def get_voice_config(language, voice_type):
    """
    Get the edge-tts voice configuration for a given language and voice type.
    Strictly returns the correct voice for the selected language — no mixing.
    """
    # Normalize voice type
    if not voice_type or voice_type == 'default':
        voice_type = 'child'  # Default to child boy voice as per requirement
    
    voice_type = voice_type.lower().strip()
    if voice_type not in ['male', 'female', 'child']:
        voice_type = 'child'
    
    # Normalize language code
    lang = language.lower().strip() if language else 'en'
    
    # Direct lookup
    if lang in EDGE_TTS_VOICES:
        config = EDGE_TTS_VOICES[lang].get(voice_type, EDGE_TTS_VOICES[lang]['default'])
        logger.info(f"✅ Voice config for {lang}/{voice_type}: {config['voice']} (pitch: {config['pitch']})")
        return config
    
    # Try base language (e.g., 'ar-ae' → 'ar')
    base_lang = lang.split('-')[0]
    if base_lang in EDGE_TTS_VOICES:
        config = EDGE_TTS_VOICES[base_lang].get(voice_type, EDGE_TTS_VOICES[base_lang]['default'])
        logger.info(f"✅ Voice config for {lang} (base: {base_lang})/{voice_type}: {config['voice']} (pitch: {config['pitch']})")
        return config
    
    # Fallback to English
    logger.warning(f"⚠️ No voice config for language '{lang}', falling back to English")
    config = EDGE_TTS_VOICES['en'].get(voice_type, EDGE_TTS_VOICES['en']['default'])
    return config


async def generate_speech_edge_tts(text, output_path, language='en', speed=1.0, pitch=1.0, voice_type='child'):
    """
    Generate speech using Microsoft Edge TTS.
    This is the PRIMARY TTS engine — high quality, multi-language, child voice support.
    
    The text is used AS-IS (no translation). Frontend sends already-translated text.
    """
    try:
        # Get voice configuration for this language + voice type
        voice_config = get_voice_config(language, voice_type)
        voice_name = voice_config['voice']
        ssml_pitch = voice_config['pitch']
        ssml_rate = voice_config['rate']
        
        # Apply user speed setting on top of voice config rate
        # Convert speed multiplier to edge-tts rate format
        if speed and speed != 1.0:
            speed_percent = int((speed - 1.0) * 100)
            ssml_rate = f"{'+' if speed_percent >= 0 else ''}{speed_percent}%"
        
        logger.info(f"═══════════════════════════════════════")
        logger.info(f"🎙️ EDGE-TTS GENERATION")
        logger.info(f"═══════════════════════════════════════")
        logger.info(f"  📝 Text: '{text}'")
        logger.info(f"  🌐 Language: {language}")
        logger.info(f"  🎤 Voice Type: {voice_type}")
        logger.info(f"  🔊 Voice Name: {voice_name}")
        logger.info(f"  🎵 SSML Pitch: {ssml_pitch}")
        logger.info(f"  ⚡ SSML Rate: {ssml_rate}")
        logger.info(f"═══════════════════════════════════════")
        
        # Generate speech with edge-tts
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_name,
            pitch=ssml_pitch,
            rate=ssml_rate
        )
        
        # Save as MP3 first (edge-tts outputs MP3)
        temp_mp3 = output_path.replace('.wav', '.mp3')
        await communicate.save(temp_mp3)
        
        logger.info(f"✅ Edge-TTS audio saved: {temp_mp3}")
        
        # Convert MP3 to WAV for consistent output format
        if PYDUB_AVAILABLE:
            try:
                sound = AudioSegment.from_mp3(temp_mp3)
                
                # Normalize audio to prevent clipping
                sound = sound.normalize()
                
                # Export as high-quality WAV
                sound.export(
                    output_path,
                    format="wav",
                    parameters=[
                        "-ar", "22050",
                        "-ac", "1",
                        "-acodec", "pcm_s16le"
                    ]
                )
                
                # Clean up temp MP3
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
                
                logger.info(f"✅ Converted to WAV: {output_path}")
                return True
                
            except Exception as conv_err:
                logger.warning(f"⚠️ WAV conversion failed: {conv_err}, using MP3 directly")
                # Keep as MP3 if conversion fails
                final_path = output_path.replace('.wav', '.mp3')
                if temp_mp3 != final_path:
                    os.rename(temp_mp3, final_path)
                return True
        else:
            # No pydub, keep as MP3
            logger.info("ℹ️ pydub not available, keeping MP3 format")
            final_path = output_path.replace('.wav', '.mp3')
            if temp_mp3 != final_path and os.path.exists(temp_mp3):
                os.rename(temp_mp3, final_path)
            return True
        
    except Exception as e:
        logger.error(f"❌ Edge-TTS generation failed: {str(e)}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False


def generate_speech_gtts_fallback(text, output_path, language='en', speed=1.0, voice_type='child'):
    """
    Fallback TTS using gTTS when edge-tts is unavailable.
    Applies pitch shifting for child voice effect using pydub.
    """
    if not GTTS_AVAILABLE:
        logger.error("❌ gTTS not available for fallback")
        return False
    
    try:
        # Map language codes for gTTS
        lang_map = {
            'en': 'en',
            'ur': 'ur',
            'ar': 'ar',
            'ar-ae': 'ar',
            'es': 'es',
            'hi': 'hi',
            'fr': 'fr',
            'de': 'de',
            'zh': 'zh-CN',
            'ja': 'ja'
        }
        
        gtts_lang = lang_map.get(language, 'en')
        logger.info(f"🔄 gTTS fallback: language={gtts_lang}, voice_type={voice_type}")
        
        # Generate with gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        temp_mp3 = output_path.replace('.wav', '_gtts_temp.mp3')
        tts.save(temp_mp3)
        
        # Apply voice effects with pydub if available
        if PYDUB_AVAILABLE:
            try:
                sound = AudioSegment.from_mp3(temp_mp3)
                
                # Apply child voice effect: raise pitch by 25%
                if voice_type == 'child':
                    logger.info("👶 Applying child voice pitch shift (gTTS fallback)")
                    new_frame_rate = int(sound.frame_rate * 1.25)
                    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                    sound = sound.set_frame_rate(44100)
                elif voice_type == 'male':
                    logger.info("👨 Applying male voice pitch shift")
                    new_frame_rate = int(sound.frame_rate * 0.85)
                    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                    sound = sound.set_frame_rate(44100)
                
                # Apply speed
                if speed and speed != 1.0:
                    new_frame_rate = int(sound.frame_rate / speed)
                    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                    sound = sound.set_frame_rate(44100)
                
                # Normalize
                sound = sound.normalize()
                
                # Export WAV
                sound.export(
                    output_path,
                    format="wav",
                    parameters=["-ar", "22050", "-ac", "1", "-acodec", "pcm_s16le"]
                )
                
                os.remove(temp_mp3)
                logger.info(f"✅ gTTS fallback audio generated: {output_path}")
                return True
                
            except Exception as pydub_err:
                logger.warning(f"⚠️ pydub processing failed in fallback: {pydub_err}")
                # Keep MP3 without processing
                final_path = output_path.replace('.wav', '.mp3')
                os.rename(temp_mp3, final_path)
                return True
        else:
            # No pydub, keep MP3
            final_path = output_path.replace('.wav', '.mp3')
            os.rename(temp_mp3, final_path)
            return True
            
    except Exception as e:
        logger.error(f"❌ gTTS fallback failed: {str(e)}")
        return False


def generate_speech(text, output_path, language='en', speed=1.0, pitch=1.0, voice_type='child'):
    """
    Generate speech using available TTS engines.
    Priority: edge-tts (primary) → gTTS (fallback) → placeholder
    
    IMPORTANT: Text is used AS-IS. No translation is performed here.
    The frontend sends already-translated text for each language.
    """
    # Normalize voice type — default to 'child' as per requirement
    if not voice_type or voice_type == 'default':
        voice_type = 'child'
    
    logger.info(f"🎙️ generate_speech called: lang={language}, voice={voice_type}, text='{text[:50]}...'")
    
    # ═══════════════════════════════════════════════════
    # PRIMARY: Edge-TTS (best quality, native multi-language, SSML pitch for child voice)
    # ═══════════════════════════════════════════════════
    try:
        # Run async edge-tts in a dedicated thread to avoid event loop conflicts
        edge_result = [False]
        edge_error = [None]
        
        def run_edge_tts():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    edge_result[0] = loop.run_until_complete(
                        generate_speech_edge_tts(text, output_path, language, speed, pitch, voice_type)
                    )
                finally:
                    loop.close()
            except Exception as e:
                edge_error[0] = e
        
        thread = threading.Thread(target=run_edge_tts)
        thread.start()
        thread.join(timeout=30)  # 30 second timeout
        
        if edge_error[0]:
            raise edge_error[0]
        
        if edge_result[0]:
            logger.info(f"✅ Speech generated successfully with Edge-TTS")
            return True
        else:
            logger.warning("⚠️ Edge-TTS returned False, trying fallback...")
            
    except Exception as edge_err:
        logger.error(f"❌ Edge-TTS exception: {str(edge_err)}")
        logger.info("🔄 Falling back to gTTS...")
    
    # ═══════════════════════════════════════════════════
    # FALLBACK: gTTS + pydub pitch shifting
    # ═══════════════════════════════════════════════════
    try:
        success = generate_speech_gtts_fallback(text, output_path, language, speed, voice_type)
        if success:
            logger.info(f"✅ Speech generated with gTTS fallback")
            return True
    except Exception as gtts_err:
        logger.error(f"❌ gTTS fallback exception: {str(gtts_err)}")
    
    # ═══════════════════════════════════════════════════
    # LAST RESORT: Placeholder audio
    # ═══════════════════════════════════════════════════
    logger.warning("⚠️ All TTS engines failed, creating placeholder audio")
    create_placeholder_audio(output_path, duration=2)
    return True


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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "tts_engine": "edge-tts",
        "fallback": "gTTS" if GTTS_AVAILABLE else "none",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/tts/synthesize', methods=['POST'])
@app.route('/synthesize', methods=['POST'])  # Backward compatibility
def synthesize_speech():
    """
    Synthesize speech from text.
    
    IMPORTANT: The text is used AS-IS for the specified language.
    No auto-translation is performed. The frontend sends pre-translated text.
    
    Expected JSON:
    {
        "text": "ٹکٹ نمبر 101 براہ کرم کاؤنٹر نمبر 5 پر تشریف لے جائیں",
        "language": "ur",
        "voice_type": "child",
        "speed": 1.0,
        "pitch": 1.0
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        voice_type = data.get('voice_type', 'child')  # Default to child
        language = data.get('language', 'en')
        speed = data.get('speed', 1.0)
        pitch = data.get('pitch', 1.0)
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # ═══════════════════════════════════════════════════
        # NO TRANSLATION — text is used exactly as received
        # The frontend sends correctly translated text
        # ═══════════════════════════════════════════════════
        
        logger.info(f"========== SYNTHESIS REQUEST ==========")
        logger.info(f"📝 Text: '{text}'")
        logger.info(f"🌐 Language: {language}")
        logger.info(f"🎤 Voice Type: {voice_type}")
        logger.info(f"⚡ Speed: {speed}")
        logger.info(f"🎵 Pitch: {pitch}")
        logger.info(f"🚫 Translation: DISABLED (text used as-is)")
        logger.info(f"=======================================")
        
        output_filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Generate speech — no translation, strict language matching
        success = generate_speech(text, output_path, language, speed, pitch, voice_type)
        
        if not success:
            return jsonify({"error": "Failed to generate speech"}), 500
        
        # Check if MP3 was created instead of WAV
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


@app.route('/api/tts/audio/<filename>', methods=['GET', 'HEAD', 'OPTIONS'])
def get_audio(filename):
    """Serve generated audio files with CORS headers"""
    try:
        # Handle OPTIONS preflight request
        if request.method == 'OPTIONS':
            response = jsonify({"status": "ok"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 200
        
        audio_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404
        
        # Determine mimetype based on extension
        mimetype = 'audio/wav'
        if filename.endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.endswith('.ogg'):
            mimetype = 'audio/ogg'
        
        # Send file with CORS headers
        response = send_file(audio_path, mimetype=mimetype)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Type'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
        
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
    """List all available voice options"""
    try:
        voices = [
            {"id": "child", "name": "👶 Child Boy Voice", "type": "child", "source": "edge-tts"},
            {"id": "male", "name": "👨 Male Voice", "type": "male", "source": "edge-tts"},
            {"id": "female", "name": "👩 Female Voice", "type": "female", "source": "edge-tts"},
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
                {"id": "child", "name": "👶 Child Boy Voice", "type": "child", "source": "fallback"},
                {"id": "male", "name": "👨 Male Voice", "type": "male", "source": "fallback"},
                {"id": "female", "name": "👩 Female Voice", "type": "female", "source": "fallback"}
            ]
        }), 200


if __name__ == '__main__':
    logger.info("═══════════════════════════════════════════════")
    logger.info("🚀 Starting Edge-TTS Voice Calling Service")
    logger.info("═══════════════════════════════════════════════")
    logger.info(f"  Primary TTS:  edge-tts (Microsoft Edge Neural TTS)")
    logger.info(f"  Fallback TTS: gTTS {'(available)' if GTTS_AVAILABLE else '(NOT available)'}")
    logger.info(f"  Audio tools:  pydub {'(available)' if PYDUB_AVAILABLE else '(NOT available)'}")
    logger.info(f"  Default voice: child boy (pitch +35%)")
    logger.info(f"  Translation:  DISABLED (frontend sends translated text)")
    logger.info("═══════════════════════════════════════════════")
    
    # Run the Flask app
    port = int(os.getenv('PORT', 2002))
    app.run(host='0.0.0.0', port=port, debug=True)
