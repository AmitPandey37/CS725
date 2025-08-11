import numpy as np
from helper import *

def movePatchOverImg(image, filter_size, apply_filter_to_patch):
    #ADD CODE HERE
    padding = filter_size//2 
    if (len(image.shape)>2):
        image = np.dot(image[..., :3],[.3, .6, .1])

    img = np.pad(image, padding, mode="constant", constant_values=0)
    output_image = np.zeros_like(image)
    print(output_image.shape)
    # print(output_image.shape[1])
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            patch = img[i: i+filter_size, j:j+filter_size]

            output_image[i,j] = apply_filter_to_patch(patch)

    return output_image

def detect_horizontal_edge(image_patch):
    #ADD CODE HERE
    filter = ([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    outputval = np.sum(filter*image_patch)
    return outputval

def detect_vertical_edge(image_patch):
    #ADD CODE HERE
    filter = ([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    outputval = np.sum(filter*image_patch)
    return outputval

def detect_all_edges(image_patch):
    #ADD CODE HERE
    filter_h = ([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    outputval_h = np.sum(filter_h*image_patch)
    filter_v = ([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    outputval_v = np.sum(filter_v*image_patch)


    return np.sqrt(outputval_h**2 + outputval_v**2)

def remove_noise(image_patch):
    #ADD CODE HERE
    outval = np.median(image_patch)
    return outval

def create_gaussian_kernel(size, sigma):
    #ADD CODE HERE
    # output_kernel = lambda x,y:(1/(2*np.pi*sigma**2))*np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2))

    # print(np.sum(output_kernel))

    output_kernel = np.zeros((size, size))
    center = (size - 1) / 2

    for x in range(size):
        for y in range(size):
            output_kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
            )

    # Normalize the kernel to sum to 1
    output_kernel /= np.sum(output_kernel)
    return output_kernel

def gaussian_blur(image_patch):
    filter = create_gaussian_kernel(25, 1)

    outputval = np.sum(filter*image_patch)
    #ADD CODE HERE
    
    return outputval

def unsharp_masking(image, scale):
    #ADD CODE HERE
    greyscale = np.dot(image[..., :3],[0.3,0.6,0.1])
    blur = movePatchOverImg(greyscale, 25, gaussian_blur)
    mask = greyscale - blur
    # output = np.clip(mask, 0, 255)
    out = mask + scale*greyscale

    return out

#TASK 1  
img=load_image("cutebird.png")
filter_size=3 #You may change this to any appropriate odd number
hori_edges = movePatchOverImg(img, filter_size, detect_horizontal_edge)
save_image("hori.png",hori_edges)
filter_size=3 #You may change this to any appropriate odd number
vert_edges = movePatchOverImg(img, filter_size, detect_vertical_edge)
save_image("vert.png",vert_edges)
filter_size=3 #You may change this to any appropriate odd number
all_edges = movePatchOverImg(img, filter_size, detect_all_edges)
save_image("alledge.png",all_edges)

#TASK 2
noisyimg=load_image("noisycutebird.png")
filter_size=3 #You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)

# #TASK 3
scale= 1 #You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
