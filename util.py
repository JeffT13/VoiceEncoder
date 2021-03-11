import os, csv, json
import numpy as np
from VoiceEncoder import audio
from VoiceEncoder.voice_encoder import VoiceEncoder #sorry for this

def getDiary(file_path):
  with open(file_path, newline='\n') as f:
    reader = csv.reader(f)
    case_diary = list(reader)
  return case_diary 

def casewrttm_to_dvec(audio_path, rttm_path, sd_path, device, rate, sr=16000, verbose=True):
  #preprocess wav file
  wav, labels, mask = audio.preprocess_wav(audio_path, rttm_path, sd_path, source_sr=sr) #labels are case preset currently
  #call model
  encoder = VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(audio_path).split('/')[-1]+"...")
  #create dvectors
  embed, splits = encoder.embed_utterance(wav, mask[-1], wav_labels=labels, sd_path=sd_path, verbose=verbose, rate=rate)
  if verbose:
    print(np.shape(embed[0]), np.shape(embed[1]), np.shape(embed[2]))
  return embed, splits, (wav, labels), mask
  
def case_to_dvec(audio_path, device, rate, sr=16000, verbose=True):
  case_size =[]
  #preprocess wav file
  wav, mask = audio.preprocess_wav(audio_path, source_sr=sr) #labels are case preset currently
  case_size = ((len(wav)/16000)/60, (len(mask)/16000)/60) 
  if verbose:
    
  #call model
  encoder = VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(audio_path).split('/')[-1]+"...")
    print((len(wav)/16000)/60, (len(mask)/16000)/60) 
  #create dvectors
  embed, info = encoder.embed_utterance(wav, mask, verbose=verbose, rate=rate)
  if verbose:
    print(np.shape(embed))
  return embed, info, case_size

