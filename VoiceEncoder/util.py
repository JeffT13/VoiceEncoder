import os, csv
from ResemblyzeLegal.VoiceEncoder import audio
from ResemblyzeLegal.VoiceEncoder import voice_encoder as VE

def casewrttm_to_dvec(audio_path, rttm_path, device, sr=16000, verbose=True):

  #file_path needs to be PosixPath(...)
  # using wav currently, not sure why we cant

  #preprocess wav file
  wav, labels, mask = audio.preprocess_wav(audio_path, rttm_path, source_sr=sr) #labels are case preset currently

  #call model
  encoder = VE.VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(audio_path).split('/')[-1]+"...")

  #create dvectors
  embed, splits = encoder.embed_utterance(wav, labels, mask[-1])

  if verbose:
    print(np.shape(embed[0]), np.shape(embed[1]), np.shape(embed[2]))
  return embed, splits, (wav, labels), mask
  
  
def case_to_dvec(audio_path, device, sr=16000, verbose=True):

  #preprocess wav file
  wav,  mask = audio.preprocess_wav(audio_path, source_sr=sr) #labels are case preset currently

  #call model
  encoder = VE.VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(audio_path).split('/')[-1]+"...")

  #create dvectors
  embed, info = encoder.embed_utterance(wav, mask)

  if verbose:
    print(np.shape(embed))
  return embed, info
  
  
  
# ------------


# write groundtruths rttm file to label lawyers as non-judges
# ex: case_docket = '18-280'
def createRTTM(case_docket, label_nonjudge = False):
  
  os.chdir('/content/drive/MyDrive/1006: Term Project/data')
  path = os.getcwd()+'/'+'{}.txt'.format(case_docket)
  txt_path = Path(path)

  timelst = []
  f = open(txt_path,'r')
  k = f.readlines()
  f.close()
  for u in k:
    t0, t1, spkr = u.split(' ')[0:3]       
    timelst.append((float(t0),float(t1),spkr))


  torttm = []
  for i, event in enumerate(timelst):
    #if labeling non-judge speakers 
    if label_nonjudge:
      if 'scotus_justice' in event[2]:
        torttm.append(' '.join(['SPEAKER {} 1'.format(case_docket), str(event[0]), str(round(event[1]-event[0], 2)), '<NA> <NA>', event[2],'<NA> <NA>']))
      else:
        torttm.append(' '.join(['SPEAKER {} 1'.format(case_docket), str(event[0]), str(round(event[1]-event[0], 2)), '<NA> <NA>', 'Non-Judge','<NA> <NA>']))
    else:
      torttm.append(' '.join(['SPEAKER {} 1'.format(case_docket), str(event[0]), str(round(event[1]-event[0], 2)), '<NA> <NA>', event[2],'<NA> <NA>']))

  if label_nonjudge:
    rttm_fpath = os.getcwd() + '/RTTMS/w_nonjudge/{}labels_w_nonjudge.rttm'
  else:
    rttm_fpath = os.getcwd() + '/RTTMS/wo_nonjudge/{}labels.rttm'

  with open(rttm_fpath.format(case_docket.replace('-','')), 'w') as filehandle:
      for listitem in torttm:
          filehandle.write('%s\n' % listitem)
          
def getDiary(file_path):
  with open(file_path, newline='\n') as f:
      reader = csv.reader(f)
      case_diary = list(reader)
  return case_diary 
  