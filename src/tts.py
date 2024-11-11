# %%
import torch
print(torch.cuda.is_available())
torch.cuda.get_device_name(torch.cuda.current_device())
# %%
from trainer import Trainer, TrainerArgs
#from trainer.logging.wandb_logger import WandbLogger
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

import sys
import os
import wandb
# %%
from trainer.logging.wandb_logger import WandbLogger
# %%
def add_artifact(self, file_or_dir, name, artifact_type, aliases=None):
    ###instead of adding artifact, do nothing###
    print(f"========Ignoring artifact: {name} {file_or_dir}========")
    return


WandbLogger.add_artifact = add_artifact
# %%
# Logging parameters
RUN_NAME = "Einhar1"
PROJECT_NAME = "Einhar" 
# %%
OUT_PATH = './exp/working/run/'
os.makedirs(OUT_PATH, exist_ok=True)
# %%
# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )
# %%
training_dir = "../Audiodata_Einhar"
# %%
OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  
START_WITH_EVAL = True  
BATCH_SIZE = 1
GRAD_ACUMM_STEPS = 252
LANGUAGE = "en"
# %%
model_args = GPTArgs(
    max_conditioning_length=143677,#the audio you will use for conditioning latents should be less than this 
    min_conditioning_length=66150,#and more than this
    debug_loading_failures=True,#this will print output to console and help you find problems in your ds
    max_wav_length=223997,#set this to >= the longest audio in your dataset  
    max_text_length=200, 
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026, 
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)
# %%
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=22050) 
# %%
SPEAKER_REFERENCE = "../Audiodata_Einhar/New_Masters_Einhar_Beastmaster_0_6.wav"
# %%
config = GPTTrainerConfig(
    run_eval=True,
    epochs = 10, # assuming you want to end training manually w/ keyboard interrupt
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""
        GPT XTTS training
        """,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=1,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=0, #consider decreasing if your jupyter env is crashing or similar
    eval_split_max_size=256, 
    print_step=50, 
    plot_step=100, 
    log_model_step=10000, 
    save_step=9999999999, #ALREADY SAVES EVERY EPOCHMaking this high on kaggle because Output dir is limited in size. I changed this to be size of training set/2 so I would effectively have a checkpoint every half epoch 
    save_n_checkpoints=4,#if you want to store multiple checkpoint rather than just 1, increase this
    save_checkpoints=False,# Making this False on kaggle because Output dir is limited
    print_eval=True,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,  
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[ 
        {
            "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "speaker_wav": SPEAKER_REFERENCE, 
            "language": LANGUAGE,
        },
        {
            "text": "This cake is great. It's so delicious and moist.",
            "speaker_wav": SPEAKER_REFERENCE,
            "language": LANGUAGE,
        },
        {
            "text": "And soon, nothing more terrible, nothing more true, and specious stuff that says no rational being can fear a thing it will not feel, not seeing that this is what we fear.",
            "speaker_wav": SPEAKER_REFERENCE,
            "language": LANGUAGE,
        }
        
    ],
) 

model = GPTTrainer.init_from_config(config)
# %%
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="../data_einhar/metadata.csv", language=LANGUAGE, path=training_dir
)
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.02)
# %%
print(eval_samples)
# %%
trainer = Trainer(
    TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=START_WITH_EVAL,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
# %%
