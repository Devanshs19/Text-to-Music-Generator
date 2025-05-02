from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64

# Ensure audio output directory exists
os.makedirs("audio_output", exist_ok=True)

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    return model

def generate_music_tensors(description, duration: int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )
    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )
    return output[0]

# Optional: Apply basic EQ filter for therapeutic sound emphasis
def apply_eq_filter(audio, sample_rate):
    audio = torchaudio.functional.highpass_biquad(audio, sample_rate, cutoff_freq=150)   # Cut low rumble
    audio = torchaudio.functional.lowpass_biquad(audio, sample_rate, cutoff_freq=5000)   # Soften highs
    return audio

def save_audio(samples: torch.Tensor):
    sample_rate = 32000
    save_path = "audio_output/"
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio = apply_eq_filter(audio, sample_rate)  # Apply frequency filtering
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon="🎵",
    page_title="Therapeutic Music Gen"
)

def main():
    st.title("🎧 Therapeutic Text to Music Generator")

    st.markdown("### About this app")
    st.write(
        "This app generates **therapeutic music** using Meta's MusicGen AI model. "
        "Choose emotional goals like **relaxation**, **anxiety relief**, or **better sleep**, "
        "and optionally customize it with your own description. 🎶"
    )

    # Emotion selection
    emotion = st.selectbox(
        "Choose your emotional goal:",
        ["Relaxation", "Reduce Anxiety", "Ease Depression", "Improve Focus", "Sleep Aid"]
    )

    # Emotion-to-music style mapping
    emotion_prompts = {
        "Relaxation": "gentle ambient music with soft piano and slow tempo",
        "Reduce Anxiety": "soothing acoustic guitar with nature sounds and slow pace",
        "Ease Depression": "uplifting melody with warm strings and soft percussion",
        "Improve Focus": "minimal electronic beat with steady rhythm and no lyrics",
        "Sleep Aid": "slow ambient textures with soft tones and no drums"
    }

    text_area = st.text_area("Describe any specific feeling, image, or instrument you'd like (optional):")
    time_slider = st.slider("Select time duration (In Seconds)", 5, 60, 15)

    # Combine emotion prompt with user input
    base_prompt = emotion_prompts[emotion]
    final_prompt = f"{base_prompt}. {text_area}" if text_area else base_prompt

    if st.button("Generate Music"):
        with st.spinner("Generating music..."):
            st.json({
                'Emotion': emotion,
                'Prompt used': final_prompt,
                'Duration (sec)': time_slider
            })

            music_tensors = generate_music_tensors(final_prompt, time_slider)
            save_audio(music_tensors)

            audio_filepath = 'audio_output/audio_0.wav'
            audio_file = open(audio_filepath, 'rb')
            audio_bytes = audio_file.read()

            st.subheader("🎵 Your Music")
            st.audio(audio_bytes)
            st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio File'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
