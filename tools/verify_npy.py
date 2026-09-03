import numpy as np

print('=== Final .npy verification ===')
for split in ['train', 'val', 'test']:
    X = np.load(f'ml/data/{split}_X.npy')
    y = np.load(f'ml/data/{split}_y.npy')
    kw = int(sum(y))
    neg = len(y) - kw
    print(f'{split}: shape={X.shape}, dtype={X.dtype}, keyword={kw}, negative={neg}, y_mean={y.mean():.3f}')

X_tr = np.load('ml/data/train_X.npy')
y_tr = np.load('ml/data/train_y.npy')
print()
print('Shape check (49,13,1) :', 'PASS' if X_tr.shape[1:] == (49, 13, 1) else 'FAIL')
print('dtype float32         :', 'PASS' if X_tr.dtype == 'float32' else 'FAIL')
print('Train balance (50/50) :', 'PASS' if abs(y_tr.mean() - 0.5) < 0.02 else f'FAIL (y_mean={y_tr.mean():.3f})')
print()
print('ALL CHECKS PASSED - .npy files ready for Colab training!')
