
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, NamedTuple, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
from numpy import ndarray
from PIL import Image


# # Input images

data_dir, extra_dir, out_dir = Path("data"), Path("extra"), Path("output")
adjust_dir = Path("output/adjust")
out_dir.mkdir(parents=True, exist_ok=True)

low_res_imgs = list(data_dir.glob("*.jpg"))
high_res_imgs = list(data_dir.glob("*.tif"))
extra_imgs = list(extra_dir.glob("*"))

print(f"number of images = {len(low_res_imgs)}")
print(f"number of images = {len(high_res_imgs)}")
print(f"number of images = {len(extra_imgs)}")


# # Aligning Channels

# ## Helper Functions

def est_channel_height(img: ndarray):
    return img.shape[0] // 3

def est_channel_width(img: ndarray):
    return img.shape[1]

def max_channel_height(img: Pic):
    return min(img.r.shape[0], img.b.shape[0], img.g.shape[0])

def max_channel_width(img: Pic):
    return min(img.r.shape[1], img.b.shape[1], img.g.shape[1])

class Displacement:
    """Start of each color as a row."""

    def __init__(self, r: Pixel = Pixel(), g: Pixel = Pixel(), b: Pixel = Pixel()):
        self.r = r
        self.g = g
        self.b = b

@dataclass
class Pixel:
    row: int = 0
    col: int = 0

@dataclass
class Offset:
    row: int = 0
    col: int = 0

@dataclass
class ChannelSize:
    h: int = 0
    w: int = 0


class Pic:
    def __init__(
        self,
        img: ndarray,
        dis: Displacement = None,
        ch_size: ChannelSize = None,
    ) -> None:
        self.img = img
        if dis is None:
            dis = Displacement()
            dis.g.row = est_channel_height(img)
            dis.r.row = est_channel_height(img) * 2
        self.dis = dis
        if ch_size is None:
            ch_size = ChannelSize()
            ch_size.h = est_channel_height(img)
            ch_size.w = est_channel_width(img)
        self.ch_size = ch_size


# ## Alignment Algorithms

# ### Basic

# Returns alignment index by simply dividing the image in 3
def align_basic(img: ndarray) -> Displacement:
    G_start = channel_height(img)  # floor division to get integer indices
    R_start = channel_height(img) * 2
    return Displacement(g=G_start, r=R_start)


# ### SSD

# Returns the ssd between matrix a and matrix b
def ssd(a: ndarray, b: ndarray) -> float:
    return np.sum((a - b) ** 2)


# ### NCC

# Returns the ncc between matrix a and matrix b
def ncc(a: ndarray, b: ndarray) -> float:
    assert(a.shape == b.shape)
    s = np.array([a[r] @ b[r] for r in range(a.shape[0])])
    return np.sum(s)


# ## Alignment Computations

# In[168]:


# Returns a sub_matrix extracted from img, returns None if out of bounds
def sub_image(
    img: ndarray,
    start: Pixel = Pixel(),
    ch_size: ChannelSize = None,
    offset: Offset = Offset(),
    pad_val: float = 0,
) -> ndarray:

    # if ch_size is not specified, initialize it
    if ch_size is None:
        ch_size = ChannelSize(h=est_channel_height(img), w=est_channel_width(img))
        
    res_h = ch_size.h - 2 * offset.row
    res_w = ch_size.w - 2 * offset.col
    R, C = img.shape
    
    # if out of bounds, return None
    if (
        start.row not in range(R)
        or start.col not in range(C)
        or start.row + res_h not in range(R)
        or start.col + res_w not in range(C)
    ):
        return None

    # copy wanted entries of img to result
    result = img[start.row : start.row + res_h, start.col : start.col + res_w]
    return result


# In[169]:


def align(
    pic: Pic,
    window: int = 20,
    offset=Offset(row=20, col=20),
    metric: Callable = ssd,
    use_min=True,
) -> Displacement:

    img = pic.img
    dis_est = pic.dis
    ch_size = pic.ch_size
    
    pad_val = 0 if use_min else np.inf
    best = min if use_min else max
    
    b = sub_image(img, dis_est.b, ch_size, offset)

    # find displacement for G channel
    score = {}
    for d in range(-window, window):
        g_est = Pixel(dis_est.g.row + d, 0)
        g = sub_image(img, g_est, ch_size, offset)
        if g is None:
            score[d] = pad_val
        else:
            score[d] = metric(b, g)

    # displacement that gives best result is the 'key' in dictionary that gives best score
    G_start = best(score)

    # find displacement for R channel
    score.clear()
    for d in range(-window, window):
        r_est = Pixel(dis_est.r.row + d, 0)
        r = sub_image(img, r_est, ch_size, offset)
        if r is None:
            score[d] = pad_val
        else:
            score[d] = metric(b, r)

    # displacement that gives best result is the 'key' in dictionary that gives best score
    R_start = best(score)

    return Displacement(g=Pixel(G_start, 0), r=Pixel(R_start, 0))


# ### Image Pyramid

# In[170]:


def pyramid(
    img: ndarray, *args
) -> Displacement:
    if img.size < 1500 * 500:
        return align(img, *args)
    im_resize = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    new_pyr = pyramid(im_resize, align_metric, *args)
    g_est, r_est = new_pyr.g, new_pyr.r
    G_start = int(np.round(g_est / im_resize.shape[0] * img.shape[0]))
    R_start = int(np.round(r_est / im_resize.shape[0] * img.shape[0]))
    return Displacement(g=G_start, r=R_start)


# # Test and Display Results

# In[171]:


# create and return channel matrices
def channels(pic: Pic) -> Tuple[ndarray, ndarray, ndarray]:
    r, g, b = pic.dis.r, pic.dis.g, pic.dis.b
    h, w = pic.ch_size.h, pic.ch_size.w
    img = pic.img
    print(r, g, b)
    B_mat = img[b.row : b.row + h, b.col: b.col + w]
    G_mat = img[g.row : g.row + h, g.col: g.col + w]
    R_mat = img[r.row : r.row + h, r.col: r.col + w]
    
    return R_mat, G_mat, B_mat


# In[172]:


def compute(input_img, out_dir:Path, algorithm, *args, show=False, adjust=False):
    # read input file
    print(input_img.name)
    img_mat = cv2.imread(str(input_img), cv2.IMREAD_GRAYSCALE)
    if show:
        plt.figure()
        plt.imshow(im, cmap=plt.get_cmap("gray"))

    # initiialize variables
    pic = Pic(img_mat)
        
    # do adjustments
    if adjust:
        # awb_grey(im, show=True)
        # awb_white(im, show=True)
        # fix_exposure(im, show=True)
        # crop_borders(im, show=True)
        pass
        
    # compute displacements
    d = align(pic, window = 20, offset=Offset(row=20, col=20), metric=ssd, use_min=True)
    #d = pyramid(im, algorithm)
    print(f'"{d.b}, {d.g}, {d.r}"')
    pic.dis = d
    
    # combine channels and display result
    R_mat, G_mat, B_mat = channels(pic)
    result = np.dstack([R_mat, G_mat, B_mat])
    if show:
        plt.figure()
        plt.title(f'"{R_mat.shape}, {G_mat.shape}, {B_mat.shape}"')
        plt.imshow(result)

    # save the images
    fname = out_dir / img.stem
    Image.fromarray(result).save(fname, "PNG")
    return result


# ## Testing low resolution images

# In[175]:


if __name__ == "__main__":
    for im in low_res_imgs:
        #compute(im, out_dir, align_basic)
        compute(im, out_dir, ssd)


# ## Testing high resolution images

# In[ ]:


if __name__ == "__main__":
    for im in high_res_imgs:
        save_n_display(im, out_dir, align_ssd)


# In[ ]:


if __name__ == "__main__":
    for im in high_res_imgs:
        save_n_display(im, out_dir, align_ssd)


# ## Testing extra images

# In[ ]:


if __name__ == "__main__":
    for im in extra_imgs:
        save_n_display(im, out_dir, align_ssd)


# # Adjustments

# ## Normalize Exposures

# In[ ]:


# takes in a matrix with values within [1,0],
# and transforms it so that the minimum value becomes 0, maximum value becomes 1
def fix_exposure(mat: ndarray, show=False) -> None:
    unit_len = np.max(mat) - np.min(mat)
    mat = (mat - np.amin(mat)) / unit_len
    if show:
        plt.imshow(mat)


# ## Crop Borders

# In[ ]:


def find_border(mat, axis):
    # zero pad mat at rightmost and bottom
    mat_padded = np.pad(mat, ((0, 1), (0, 1), "constant", 0))
    
    val_r, val_c = [], []
    # find row border
    for i in range[mat.shape[0]]:
        val_r[i, i + 1] = mat_padded[i] @ mat_padded[i + 1]
    r_cutoff = np.unravel_index(np.argmin(val))[1]
    # find col border
    for i in range[mat.shape[1]]:
        val_c[i, i + 1] = mat_padded[:, [i]] @ mat_padded[:, [i + 1]]
    c_cutoff = np.unravel_index(np.argmin(val))[1]
    return [r_cutoff, c_cutoff]

def crop_borders(mat, show=False):
    find_border(mat)
    if show:
        plt.imshow(balanced_im)


# ## Auto White Balance (AWB)

# In[ ]:


# Automatic (AWB)
# • Grey World: force average color of scene to grey
# • White World: force brightest object to white

def awb_grey(im, show=False):
    # Compute the mean color over the entire image
    avg_color = np.mean(im)

    # Scale the averge color to be grey (0.5)
    scaling = 0.5 / avg_color

    # Apply the scaling to the entire image
    balanced_im = im * scaling
    if show:
        plt.imshow(balanced_im)
    im = balanced_im

def awb_white(im, show=False):
    # Compute the brightest color over the entire image
    brightest_color = np.amax(im)

    # Scale the brightest color to be white (1.0)
    scaling = 1.0 / brightest_color

    # Apply the scaling to the entire image
    balanced_im = im * scaling
    if show:
        plt.imshow(balanced_im)
    im = balanced_im


# ## Apply Adjustments

# In[ ]:


# im = low_res_imgs[:1]
# plt.show(im)

# awb_grey(im, show=True)
# #awb_white(im, show=True)
# fix_exposure(im, show=True)
# #crop_borders(im, show=True)

# im_aligned = save_n_display(input, out_dir, align_ssd)


# In[ ]:


im_aligned = save_n_display(low_res_imgs[:1], out_dir, align_basic)
#awb_grey(im_aligned, show=True)
awb_white(im_aligned, show=True)
fix_exposure(im_aligned, show=True)
#crop_borders(im_aligned, show=True)


# In[ ]:




