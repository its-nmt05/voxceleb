import os
import torch
import torchaudio.transforms as T
# from codec_wrappers import EncodecWrapper, DacWrapper
from utils import get_wav_files, get_model, preprocess, pad_or_trim_to_length, save_audio
from tqdm import tqdm 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = 'voxceleb1'
MODEL = 'encodec'
# load model
model, model_sampling_rate, bands = get_model(MODEL)
model = model.to(device)
model.eval()
audio_files = get_wav_files(DATASET_PATH)

sr = 16000
resampler = T.Resample(sr, model_sampling_rate)
resampler = resampler.to(device)

for band in bands:
    for file_path in tqdm(audio_files, desc=f'Processing band {band}'):
        input = preprocess(file_path, model_sampling_rate, resample_obj=resampler, device=device)
        target_length = input.shape[-1]
        outputs = model(input, bandwidth=band).audio_values
        
        if len(outputs.shape) == 3:
            outputs = outputs.squeeze(0)
    
        outputs = pad_or_trim_to_length(outputs, target_length)
        save_path = f"voxceleb1_{MODEL}_samples/{band}"
        new_path = file_path.replace("voxceleb1", save_path) 
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        save_audio(outputs.detach().cpu(), model_sampling_rate, new_path)
        
        # Clear GPU memory
        del input
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        
