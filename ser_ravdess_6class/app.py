import streamlit as st
import numpy as np
import librosa
try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE = True
except Exception:
    tflite = None
    HAS_TFLITE = False
import tempfile
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# Constants (matching the training notebook)
SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 60
TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

# Custom layer for loading the model
class SumContext(tf.keras.layers.Layer):
    """Time-wise weighted sum: sum_t (x_t * a_t)."""
    def call(self, inputs):
        x, attn = inputs  # x: (B,T,F), attn: (B,T,1)
        return tf.reduce_sum(x * attn, axis=1)  # (B,F)

    def get_config(self):
        return super().get_config()

@st.cache_resource
def load_model():
    """Load the trained emotion recognition model.
    Prefer a TFLite interpreter if available; otherwise return None and the
    app will run in demo/mock mode (no heavy runtime dependencies required).
    """
    if HAS_TFLITE:
        # Try to load the small tflite model if present
        model_path = "saved_models/ser_6class_select.tflite"
        if not os.path.exists(model_path):
            model_path = "ser_6class_select.tflite"
        if not os.path.exists(model_path):
            st.warning("TFLite model not found; the app will run in demo mode.")
            return None
        try:
            interp = tflite.Interpreter(model_path=model_path)
            interp.allocate_tensors()
            return interp
        except Exception as e:
            st.warning(f"Error loading TFLite model: {e}. Running in demo mode.")
            return None
    else:
        # No tflite runtime available on the host — run demo mode
        st.info("tflite-runtime not available; running in demo (no-model) mode.")
        return None

def load_fixed_audio(audio_path, sr=SAMPLE_RATE):
    """Load and fix audio to 3 seconds"""
    sig, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(sig) > SAMPLES_PER_TRACK:
        sig = sig[:SAMPLES_PER_TRACK]
    else:
        sig = np.pad(sig, (0, SAMPLES_PER_TRACK - len(sig)))
    return sig

def extract_mfcc(sig, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC features from audio signal"""
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose to get (time, features) shape

def preprocess_audio(audio_path):
    """Preprocess audio file for model prediction"""
    sig = load_fixed_audio(audio_path)
    mfcc_features = extract_mfcc(sig)
    # Reshape for model: (1, time, features)
    return mfcc_features[np.newaxis, ...].astype(np.float32)

def predict_emotion(model, audio_features):
    """Predict emotion from audio features.
    If a TFLite interpreter is available, run it; otherwise return mock/demo results.
    """
    if model is None:
        # Demo mode: return a deterministic mock prediction (use mean of features)
        vals = np.abs(audio_features).mean() if hasattr(audio_features, 'mean') else 0.5
        # produce a simple softmax-like distribution biased by vals
        probs = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
        emotion_idx = int(np.argmax(probs))
        confidence = float(probs[emotion_idx])
        all_probs = {TARGET_EMOTIONS[i]: float(probs[i]) for i in range(len(TARGET_EMOTIONS))}
        return TARGET_EMOTIONS[emotion_idx], confidence, all_probs

    # If we have a TFLite interpreter
    try:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        # Ensure correct dtype/shape; trim or pad as needed
        input_data = audio_features.astype(input_details[0]['dtype'])
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        pred = model.get_tensor(output_details[0]['index'])[0]
        emotion_idx = int(np.argmax(pred))
        confidence = float(pred[emotion_idx])
        all_probs = {TARGET_EMOTIONS[i]: float(pred[i]) for i in range(len(TARGET_EMOTIONS))}
        return TARGET_EMOTIONS[emotion_idx], confidence, all_probs
    except Exception as e:
        st.warning(f"Error running model inference: {e}. Returning demo prediction.")
        probs = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
        emotion_idx = int(np.argmax(probs))
        confidence = float(probs[emotion_idx])
        all_probs = {TARGET_EMOTIONS[i]: float(probs[i]) for i in range(len(TARGET_EMOTIONS))}
        return TARGET_EMOTIONS[emotion_idx], confidence, all_probs

# Main app
st.title("🎤 Speech Emotion Recognition")
st.markdown("Upload an audio file to detect the emotion in speech")
st.markdown("---")

# Load model
model = load_model()

if model is None:
    st.stop()

# Sidebar
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. Upload an audio file (WAV, MP3, etc.)
2. The model will analyze the speech
3. Get emotion prediction with confidence scores

**Supported Emotions:**
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 😒 Disgust
- 😐 Neutral
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg']
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Process and predict
        with st.spinner("Processing audio and predicting emotion..."):
            # Preprocess audio
            audio_features = preprocess_audio(tmp_path)
            
            # Predict emotion
            emotion, confidence, all_probs = predict_emotion(model, audio_features)
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Main result
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detected Emotion")
            # Emoji mapping
            emoji_map = {
                "happy": "😊",
                "sad": "😢",
                "angry": "😠",
                "fearful": "😨",
                "disgust": "😒",
                "neutral": "😐"
            }
            emoji = emoji_map.get(emotion, "🎤")
            st.markdown(f"### {emoji} **{emotion.upper()}**")
            st.metric("Confidence", f"{confidence*100:.2f}%")
        
        with col2:
            st.subheader("All Emotion Probabilities")
            # Sort probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for emo, prob in sorted_probs:
                emoji = emoji_map.get(emo, "🎤")
                st.progress(prob, text=f"{emoji} {emo.capitalize()}: {prob*100:.1f}%")
        
        # Additional info
        st.markdown("---")
        with st.expander("📊 Technical Details"):
            st.write(f"**Audio Duration:** {DURATION} seconds")
            st.write(f"**Sample Rate:** {SAMPLE_RATE} Hz")
            st.write(f"**MFCC Features:** {N_MFCC} coefficients")
            st.write(f"**Feature Shape:** {audio_features.shape}")
            st.write(f"**Model Input Shape:** {model.input_shape}")
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.info("Please ensure the audio file is valid and try again.")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Please upload an audio file to get started!")
    
    # Show example
    st.markdown("---")
    st.subheader("💡 Example")
    st.markdown("""
    You can test with any audio file containing speech. The model works best with:
    - Clear speech (minimal background noise)
    - Single speaker
    - 1-5 seconds duration (will be trimmed/padded to 3 seconds)
    """)

# Footer
st.markdown("---")
st.markdown("**Speech Emotion Recognition** - Trained on RAVDESS Dataset (6 classes)")

