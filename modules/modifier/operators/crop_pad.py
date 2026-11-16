"""Explicit whole-image crop/pad geometry, not a face-pose edit."""
import numpy as np
from PIL import Image,ImageOps
from modules.modifier.operators.base import ImageTransform
from shared.contracts import ModuleFailure


def apply(image,parameters,max_pixels):
    left,top,right,bottom=parameters.crop_box or [0,0,image.width,image.height]
    if not (0<=left<right<=image.width and 0<=top<bottom<=image.height):
        raise ModuleFailure('invalid_parameters','Crop box must be inside the input canvas',field='modifier.parameters.crop_box')
    pad_left,pad_top,pad_right,pad_bottom=parameters.padding
    size=(right-left+pad_left+pad_right,bottom-top+pad_top+pad_bottom)
    if size[0]*size[1]>max_pixels:
        raise ModuleFailure('image_too_large','Crop/pad output exceeds the configured pixel limit')
    cropped=image.crop((left,top,right,bottom))
    if parameters.fill_mode=='constant':
        output=ImageOps.expand(cropped,border=tuple(parameters.padding),fill=tuple(parameters.fill))
    else:
        output=Image.fromarray(np.pad(np.asarray(cropped),((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),mode='edge'))
    return ImageTransform(output,details=dict(crop_box=[left,top,right,bottom],
        crop_coordinates='left/top inclusive; right/bottom exclusive; original canvas pixels',
        padding_ltrb=parameters.padding,fill_mode=parameters.fill_mode,fill_rgb=parameters.fill,
        fill_rgb_used=parameters.fill_mode=='constant',output_size=list(size),
        input_to_output=[[1,0,pad_left-left],[0,1,pad_top-top],[0,0,1]],
        operation='whole_canvas_geometry',face_pose_changed='not_evaluated'))
