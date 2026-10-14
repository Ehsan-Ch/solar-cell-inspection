import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

def resize(image):
    arr = np.asarray(image)
    if arr.ndim != 2 or min(arr.shape) < 16 or not np.isfinite(arr).all():
        raise ValueError('Expected a finite grayscale image at least 16 x 16')
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError('Expected 8-bit intensity range')
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((96, 96), Image.Resampling.BILINEAR))

def extract(image):
    a = resize(image)
    x = a.astype(float) / 255
    # Absolute intensity is retained; there is no per-image contrast normalization.
    stats = [x.mean(), x.std(), *np.quantile(x, [.05, .25, .5, .75, .95]),
             np.mean(x < .1), np.mean(x < .25), np.mean(x > .9)]
    tiles = [x[y:y+24, z:z+24] for y in range(0, 96, 24) for z in range(0, 96, 24)]
    basic = np.array(stats + [v.mean() for v in tiles] + [v.std() for v in tiles])
    gradients = hog(x, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
    lbp = local_binary_pattern(a, P=8, R=1, method='uniform')
    texture = np.concatenate([np.bincount(lbp[y:y+32,z:z+32].astype(int).ravel(), minlength=10)/1024
                              for y in range(0, 96, 32) for z in range(0, 96, 32)])
    return np.concatenate([basic, gradients, texture])

def batch(images):
    return np.array([extract(im) for im in images])

def occlusion(image, predict, grid=6):
    """Score drop after mean-filling each patch; sensitivity, not segmentation."""
    a = resize(image)
    base = float(predict(batch([a]))[0])
    patches = []
    for row in range(grid):
        for col in range(grid):
            b = a.copy()
            b[row*96//grid:(row+1)*96//grid,col*96//grid:(col+1)*96//grid] = round(float(a.mean()))
            patches.append(b)
    return (base - np.asarray(predict(batch(patches)))).reshape(grid, grid)
