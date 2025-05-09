import torch
import torchaudio
import csv
from tqdm import tqdm
from speechbrain.inference.speaker import SpeakerRecognition

VERIFY_FILE = r"veri_test.txt"
ORIG_DIR = r"voxceleb1"
CODEC_DIR = r"voxceleb1_encodec_samples"
CSV_FILE = r"verification_results.csv"
bands = [1.5, 3.0, 6.0, 12.0, 24.0] # supported bands
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

lines = open(VERIFY_FILE).readlines()
with open(CSV_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["band", "file1", "file2", "label", "score", "prediction"])
    
    for band in bands:
        for line in tqdm(lines, desc=f"Processing Band {band}", ncols=80):
            label, file1, file2 = line.strip().split()
            path1 = f"{CODEC_DIR}/{band}/{file1}"
            path2 = f"{CODEC_DIR}/{band}/{file2}"
            wav1, _ = torchaudio.load(path1)
            wav2, _ = torchaudio.load(path2)
            score, pred = model.verify_batch(wav1, wav2)
            writer.writerow([band, file1, file2, label, f"{score.item():.4f}", int(pred)])
