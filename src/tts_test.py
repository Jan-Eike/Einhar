# %%
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf  # To save the output as a wav file

# Step 1: Load the model configuration
config = XttsConfig()
config.load_json("./exp/working/run/Einhar1-November-10-2024_04+42AM-8030d20/config.json")

# Step 2: Initialize the model
model = Xtts.init_from_config(config)

# Step 3: Load the pre-trained weights
model.load_checkpoint(config, checkpoint_dir="./exp/working/run/Einhar1-November-10-2024_06+44AM-8030d20/", vocab_path="./exp/working/run/XTTS_v2.0_original_model_files/vocab.json", eval=True)
#model.load_checkpoint(config, checkpoint_dir="./exp/working/run/XTTS_v2.0_original_model_files/", vocab_path="./exp/working/run/XTTS_v2.0_original_model_files/vocab.json", eval=True)

# Optional: If you have CUDA installed and want to use GPU, uncomment the line below
model.cuda()
print(model)
# Step 4: Synthesize the output
outputs = model.synthesize(
    "Do you not have nets, exile?",
    config,
    speaker_wav="../Audiodata_Einhar/New_Masters_Einhar_Beastmaster_0_6.wav",  # Replace with the correct path
    gpt_cond_len=3,
    language="en",
)

# Step 5: Save the synthesized speech to a wav file
output_wav = outputs['wav']
sf.write('output.wav', output_wav, config.audio.sample_rate)

print("Speech synthesis complete and saved to output.wav")
# %%
