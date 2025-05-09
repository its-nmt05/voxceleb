import torch
import torchaudio
import torchaudio.functional as F
import os
import csv
import json
import errno

def enumerate_csv_rows(filename):
    """
    Reads a CSV file row by row and yields each row as a dictionary,
    using the header row as keys.

    Args:
        filename (str): The path to the CSV file.

    Yields:
        dict: A dictionary representing a row from the CSV file,
            where keys are from the header row.
    """
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                yield row
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")
def create_symlink(filepath, target_dir):
    """
    Creates a symbolic link to a file in a specified directory.

    Args:
        filepath (str): The path to the original file.
        target_dir (str): The directory where the symbolic link should be created.

    Returns:
        str: The path to the created symbolic link, or None on error.
              Returns the symlink path even if the symlink already exists.

    Raises:
        TypeError: If either `filepath` or `target_dir` is not a string.
        OSError:  If the file exists and is not a symbolic link, or
                   if another error occurs during symlink creation
                   (e.g., permissions, invalid path).  The OSError's
                   errno attribute will indicate the specific error.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
    if not isinstance(target_dir, str):
        raise TypeError("target_dir must be a string")

    # Extract the filename from the filepath
    filename = os.path.basename(filepath)
    # Construct the full path for the symbolic link
    symlink_path = os.path.join(target_dir, filename)

    # Check if the target directory exists.
    if not os.path.exists(target_dir):
        raise OSError(errno.ENOENT, "Target directory does not exist")

    try:
        # Create the symbolic link
        os.symlink(filepath, symlink_path)
        return symlink_path  # Return the path to the symlink

    except OSError as e:
        if e.errno == errno.EEXIST:
            if os.path.islink(symlink_path):
                return symlink_path # symlink already exists, return the path
            else:
                raise OSError(errno.EEXIST, "File exists and is not a symbolic link") from e
        else:
            raise  # Re-raise the original OSError for other errors

def get_model(model):

    if "encodec" in model.lower():
        # model_id = "facebook/encodec_24khz"
        # model = EncodecModel.from_pretrained(model_id)
        from transformers import EncodecModel
        # model = EncodecModel.from_pretrained("/home/jyip/.cache/huggingface/hub/models--facebook--encodec_24khz/snapshots/c1dbe2ae3f1de713481a3b3e7c47f357092ee040", local_files_only=True)
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        model_sampling_rate = model.config.sampling_rate
        bands = model.config.target_bandwidths
        
    elif "dac" in model.lower():
        from transformers import DacModel
        model = DacModel.from_pretrained("descript/dac_16khz", local_files_only=True )

        #cursed monkey patch
        # Store the original encode method
        original_quantizer_encode = model.quantizer.forward

        # Create a new function that ignores the bandwidth parameter
        def patched_encode(input_tensor, bandwidth=None, *args, **kwargs):
            # Simply call the original encode method without the bandwidth parameter
            quantized_representation, audio_codes, projected_latents, commitment_loss, codebook_loss =  original_quantizer_encode(input_tensor, *args, **kwargs)
            return audio_codes

        # Replace the original method with our patched version
        model.quantizer.encode = patched_encode

        # #patched decode function
        def patched_decode(audio_codes):
            # Simply call the original encode method without the bandwidth parameter
            return model.quantizer.from_codes(audio_codes)[0]
        model.quantizer.decode = patched_decode

        # Store the original forward method
        original_forward = model.forward

        # Create a new function that ignores the bandwidth parameter
        def patched_forward(input_tensor, bandwidth=None, *args, **kwargs):
            # Call the original forward method without the bandwidth parameter
            return original_forward(input_tensor, *args, **kwargs)

        # Replace the original method with our patched version
        model.forward = patched_forward

        model_sampling_rate = model.config.sampling_rate
        bands = ['8.0']
    else:
        NotImplementedError
    
    return model, model_sampling_rate, bands

def get_class_labels(filename):
    with open(filename,'r') as json_file:
        data = json.load(json_file)
    return data
def create_empty_csv(filename, column_names):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

# def add_row_to_csv(filename, row_dict):
#     with open(filename, 'a', newline='') as csvfile:
#         fieldnames = list(row_dict.keys())
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writerow(row_dict)

def add_row_to_csv(filename, row_dict):
    # First, read the existing headers from the CSV to maintain order
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        fieldnames = next(reader)  # Get the headers in their original order
    
    # Then append the row using those ordered fieldnames
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row_dict)

def save_audio(audio_data, sample_rate, filename):
    """
    Save audio data as a WAV file using torchaudio.
    
    Args:
        audio_data: numpy array or tensor of audio samples
        sample_rate: sampling rate in Hz
        filename: output filename
    """
    # Convert to tensor if numpy array
    if not torch.is_tensor(audio_data):
        audio_data = torch.from_numpy(audio_data).to(torch.float32)
    
    # Ensure audio is 2D (channels, samples)
    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)
    
    # Ensure values are in [-1, 1]
    if audio_data.max() > 1 or audio_data.min() < -1:
        audio_data = torch.clamp(audio_data, -1, 1)
    
    # Save as WAV
    torchaudio.save(filename, audio_data, sample_rate)

def find_tgt_2mix_wav_files(directory):
    return [
        file for file in os.listdir(directory) 
        if "tgt" in file and file.endswith('.wav') and file.count('_') == 2
    ]

def find_tgt_wav_files(directory):
    return [
        file for file in os.listdir(directory) 
        if "tgt" in file and file.endswith('.wav') and file.count('_') == 1
    ]

def find_all_wav_files(directory):
    return [
        file for file in os.listdir(directory) 
        if file.endswith('.wav')
    ]

def get_wav_files(directory):
    """
    Gets the full path of all .wav files in the specified directory.
    
    Args:
        directory (str): Path to the directory to search in
    
    Returns:
        list: A list of full paths to all .wav files in the directory
    """
    wav_files = []
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return wav_files
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a .wav extension (case insensitive)
            if file.lower().endswith('.wav'):
                # Get the full path and add it to the list
                full_path = os.path.join(root, file)
                wav_files.append(full_path)
    
    return wav_files

def preprocess(filepath, model_sampling_rate, resample_obj=None, device=None):
    # Default to CPU if no device specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loading audio file and moving to device
    waveform, sample_rate = torchaudio.load(filepath)
    waveform = waveform[0,:].unsqueeze(0).to(device)

    #preprocessing
    if sample_rate != model_sampling_rate:
        if not resample_obj:
            waveform = F.resample(waveform, sample_rate, new_freq=model_sampling_rate)
        else:
            waveform = resample_obj(waveform)

    return waveform.unsqueeze(0) if len(waveform.shape) == 2 else waveform


def pad_or_trim_to_length(signal, target_length, pad_end=True):
    """
    Pad or trim a signal to match the target length.
    Works with [C,T] dimension data where C is channels and T is time.
    
    Args:
        signal (torch.Tensor): Signal with shape [C,T]
        target_length (int): Target length for time dimension
        pad_end (bool): Whether to pad at the end (True) or beginning (False)
        
    Returns:
        torch.Tensor: Padded or trimmed signal of shape [C,target_length]
    """
    signal_length = signal.size(1)  # Time is now dimension 1
    
    if signal_length > target_length:
        # Trim in time dimension
        return signal[:, :target_length]
    elif signal_length < target_length:
        # Pad in time dimension
        pad_size = target_length - signal_length
        padded = torch.zeros(signal.size(0), target_length, 
                            device=signal.device, dtype=signal.dtype)
        
        if pad_end:
            padded[:, :signal_length] = signal
        else:
            padded[:, -signal_length:] = signal
        
        return padded
    else:
        # Already the correct length
        return signal

def align_estimated_to_reference(reference, estimated, max_shift=100):
    """
    Align estimated signal to reference by shifting only the estimated signal.
    Reference signal remains unchanged and unpadded.
    Works with [C,T] dimension data.
    
    Args:
        reference (torch.Tensor): Reference audio signal [B,C,T] or [C,T]
        estimated (torch.Tensor): Estimated audio signal [B,C,T] or [C,T]
        max_shift (int): Maximum allowed shift in either direction
        
    Returns:
        torch.Tensor: Aligned estimated signal, trimmed to match reference length
    """
    # Check if input has batch dimension
    reference_was_2d = reference.dim() == 2  # [C,T]
    estimated_was_2d = estimated.dim() == 2  # [C,T]
    
    if reference_was_2d:
        reference = reference.unsqueeze(0)  # [1,C,T]
    if estimated_was_2d:
        estimated = estimated.unsqueeze(0)  # [1,C,T]
    
    batch_size = reference.shape[0]
    aligned_ests = []
    
    for b in range(batch_size):
        ref = reference[b]  # [C,T]
        est = estimated[b]  # [C,T]
        
        # We'll use the first channel for alignment calculation
        ref_channel = ref[0]  # [T]
        est_channel = est[0]  # [T]
        
        # Calculate cross-correlation using convolution
        pad_size = min(max_shift, ref_channel.size(0), est_channel.size(0))
        
        # Pad only the estimated signal for correlation calculation
        flipped_est = est_channel.flip(0)
        padded_est = torch.zeros(flipped_est.size(0) + pad_size, 
                                device=flipped_est.device, dtype=flipped_est.dtype)
        padded_est[:flipped_est.size(0)] = flipped_est
        
        # Use reference without padding for correlation
        correlation = torch.nn.functional.conv1d(
            padded_est.unsqueeze(0).unsqueeze(0),
            ref_channel.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        # Find max correlation index
        max_idx = torch.argmax(correlation)
        # Calculate shift considering we didn't pad the reference
        shift = max_idx - (ref_channel.size(0) - 1)
        
        # Shift estimated signal based on correlation
        channels = est.size(0)
        
        if shift >= 0:
            # Need to pad estimated at the beginning
            aligned_est = torch.zeros(channels, est.size(1) + shift, 
                                    device=est.device, dtype=est.dtype)
            aligned_est[:, shift:] = est
        else:
            # Estimated signal starts too early, trim the beginning
            aligned_est = est[:, -shift:]
        
        # Match the length to reference
        ref_length = ref.size(1)
        aligned_est = pad_or_trim_to_length(aligned_est, ref_length)
        
        aligned_ests.append(aligned_est)
    
    # Stack results and maintain original dimensions
    if batch_size == 1 and estimated_was_2d:
        return aligned_ests[0]
    else:
        return torch.stack(aligned_ests)