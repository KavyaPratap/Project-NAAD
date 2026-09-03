import os
import glob

# Count raw_wav (TTS generated)
raw_pos = glob.glob('raw_wav/positive/*.wav')
raw_neg = glob.glob('raw_wav/negative/*.wav')

# Count real_data_wav
real = glob.glob('real_data_wav/*.wav')

print('=== raw_wav (TTS-generated keyword WAVs) ===')
print(f'  positive (keyword): {len(raw_pos)} files')
print(f'  negative          : {len(raw_neg)} files')

print()
print('=== real_data_wav ===')
print(f'  files: {len(real)}')
if real:
    print(f'  samples: {[os.path.basename(f) for f in real[:5]]}')

print()
print('=== current ml/data (destination) ===')
for split in ['train', 'val', 'test']:
    for cls in ['keyword', 'negative']:
        p = os.path.join('ml', 'data', split, cls)
        wavs = glob.glob(os.path.join(p, '*.wav'))
        print(f'  {split}/{cls}: {len(wavs)} files')
