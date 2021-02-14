import os, csv
from resemblyzer import audio
'''

git submodule check (GSM)

'''

def case_to_dvec(file_path, casediary, device = device, sr = sr, verbose=True):

  #file_path needs to be PosixPath(...)
  # using wav currently, not sure why we cant

  #preprocess wav file
  wav, labels, mask = audio.preprocess_wav(file_path, source_sr=16000, casediary = casediary) #labels are case preset currently

  #call model
  encoder = VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(file_path).split('/')[-1]+"...")

  #create dvectors
  embed, splits = encoder.embed_utterance(wav, labels, mask[-1])

  if verbose:
    print(np.shape(embed[0]), np.shape(embed[1]), np.shape(embed[2]))
  return embed, splits, (wav, labels), mask
  
  
  
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
  
  
#NEW FUNCTIONS
def label_wav(wav_len, sr, casetimes, spkr):
  mask = np.zeros(wav_len)
  st = 0
  for entry in casetimes:
    temp = entry[0].split(' ')
    time, spk = temp[4], temp[7]
    idx = int(float(time)*sr)+st
    if idx<wav_len:
      mask[st:idx] = spkr[spk]
    else:
      mask[st:]=spkr[spk]
    st = idx
  return mask

def wav_label_for_melspec(wav, labels, hop=160, window=400, overlap_label=999):
  mel_lab = np.zeros(int(len(wav)/hop) + 1)
  for i  in range(len(mel_lab)):
    idx = (i*hop)
    lab = np.array(labels[idx:idx+window])
    if len(np.unique(lab))==1:
      mel_lab[i]=lab[0]
    else:
      mel_lab[i]=overlap_label
  return mel_lab