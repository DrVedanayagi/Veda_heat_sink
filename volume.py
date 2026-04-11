import SimpleITK as sitk
import numpy as np
import sys
import shutil
import copy
import os

from misc import utils


class Volume:
    def __init__(self, filename):
        self.filename = filename

        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        self.array = sitk.GetArrayFromImage(itkimage)
        self.array = np.swapaxes(self.array, 0, 2)

        # Read the origin of the volume, will be used to convert the coordinates from world to voxel and vice versa.
        self.origin = np.array(itkimage.GetOrigin())

        # Read the spacing along each dimension
        self.spacing = np.array(itkimage.GetSpacing())

        self.dim_size = np.array(np.shape(self.array))

        self.direction = np.array(itkimage.GetDirection()).reshape((3, 3))

        self.model_mat = np.zeros((4,4))
        self.model_mat[:3,:3] = self.direction
        self.model_mat[:3,3] = self.origin
        self.model_mat[3,3] = 1
        self.model_mat[:3,:3] *= self.spacing * self.dim_size

        self.intercept = 0.
        self.slope = 1.

        # Read fields not exposed or handled through SimpleITK
        mhd_fields = {}
        if os.path.isfile(filename) and os.path.splitext(filename)[-1].lower() == '.mhd':
            with open(filename) as vol_header:
                for line in vol_header:
                    name, val = line.partition("=")[::2]
                    mhd_fields[name.strip()] = val.strip()

        if 'InterceptSlope' in mhd_fields:
            vals = list(map(float, mhd_fields['InterceptSlope'].split()))
            if len(vals) == 2:
                self.intercept = vals[0]
                self.slope = vals[1]

    def rescale_voxel_value (self, val):
        rescaledVal = val
        if self.slope != 0.:
            rescaledVal = self.intercept + val * self.slope
        return rescaledVal

    def unrescale_voxel_value (self, val):
        original = val
        if self.slope != 0.:
            original = (val - self.intercept) / self.slope
        return original

    def save_image (self, filename):
        new_itk_image = sitk.GetImageFromArray(np.swapaxes(self.array, 0, 2))
        new_itk_image.SetOrigin(self.origin)
        new_itk_image.SetSpacing(self.spacing)
        new_itk_image.SetDirection(self.direction.flatten())

        sitk.WriteImage(new_itk_image, filename)
        
    def save_image_hdr(self, filename):
        img = sitk.GetImageFromArray(np.swapaxes(self.array, 0, 2))
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        img.SetDirection(self.direction.flatten())

        filename_base, filename_ext = os.path.splitext(filename)
        sitk.WriteImage(img, filename_base + '.mhd')
        
        if os.path.exists(filename_base + '.hdr'):
            os.remove(filename_base + '.hdr')
        os.rename(filename_base + '.mhd', filename_base + '.hdr')
        
        if os.path.exists(filename_base + '.img'):
            os.remove(filename_base + '.img')
        os.rename(filename_base + '.raw', filename_base + '.img')
        
        byte_LUT = {'16-bit unsigned integer': (2, 0, 65535),
                    '8-bit unsigned integer': (1, 0, 255),
                    '16-bit signed integer': (2, -32768, 32767),
                    '8-bit signed integer': (1, -128, 127)
                    }
        with open(filename_base + '.hdr', 'w') as f:
            f.write('%d %d %d\n%f %f %f\n%f %f\n%d\n' % (*img.GetSize(), *img.GetSpacing(), -1024, 1, 0))
            f.write('%f %f %f %f\n' % (*img.GetDirection()[:3], img.GetOrigin()[0]))
            f.write('%f %f %f %f\n' % (*img.GetDirection()[3:6], img.GetOrigin()[1]))
            f.write('%f %f %f %f\n' % (*img.GetDirection()[6:9], img.GetOrigin()[2]))
            f.write('%d\n%d %d\n' % byte_LUT[img.GetPixelIDTypeAsString()])

    def split_as_multilabel(self):
        masks = {}

        for i in range (1, np.amax(self.array)+1):
            mask = np.array(self.array == i).astype(np.uint8)
            if np.amax(mask) > 0:
                masks[i] = copy.deepcopy(self)
                masks[i].array = mask

        return masks
    
    def split_as_bitmask(self):
        masks = {}

        for i in range (1, 17):
            mask = np.array(self.array & (1 << (i-1)) > 0).astype(np.uint8)
            if np.amax(mask) > 0:
                masks[i] = copy.deepcopy(self)
                masks[i].array = mask

        return masks  


    def __str__(self): 
        return 'origin= ' +  str(self.origin) + '\nspacing= ' + str(self.spacing) + '\ndim_size= ' + str(self.dim_size) + '\ndirection=\n' + str(self.direction) + '\nmodel matrix=\n' + str(self.model_mat)


    # Returns the bounding box in world coordinates
    def get_box(self, labelid = -1):
        if np.count_nonzero(self.array) == 0:
            print("Error on bounding box: volume data is empty", file=sys.stderr)
            box_MS = {'Xmin': 0, 'Xmax': 1,
                      'Ymin': 0, 'Ymax': 1,
                      'Zmin': 0, 'Zmax': 1}
        else:
            if labelid == -1:
                indices = np.where(self.array != 0)
            else:
                indices = np.where(self.array == labelid)

            box_MS = {'Xmin': min(indices[0]) / self.dim_size[0],
                      'Xmax': max(indices[0]) / self.dim_size[0],
                      'Ymin': min(indices[1]) / self.dim_size[1],
                      'Ymax': max(indices[1]) / self.dim_size[1],
                      'Zmin': min(indices[2]) / self.dim_size[2],
                      'Zmax': max(indices[2]) / self.dim_size[2]}
        box_WS = utils.MS_to_WS_box(box_MS, self.model_mat)

        return box_WS

def combine_bitmasks (masks):
        
    if len(masks) > 0:
        bitmask = copy.deepcopy (masks[next(iter(masks))])
        bitmask.array = np.zeros(bitmask.array.shape,dtype=np.uint16)
        for mask_id in masks:
            bitmask.array += np.array((1 << (mask_id-1)) * (masks[mask_id].array > 0)).astype(np.uint16)

        return bitmask

    return None

def combine_multilabel (masks, prioritties_map):
        
    if len(masks) > 0:
        multilabel = copy.deepcopy (masks[next(iter(masks))])
        multilabel.array = np.zeros(multilabel.array.shape,dtype=np.uint8)
        saved_priority = np.zeros(multilabel.array.shape,dtype=np.uint8)
        for mask_id in masks:
            current_priority = 0
            if mask_id in prioritties_map:
                current_priority = prioritties_map[mask_id]

            multilabel.array = np.where ((masks[mask_id].array > 0) * current_priority > saved_priority, mask_id, multilabel.array).astype(np.uint8)
            saved_priority = np.where ((masks[mask_id].array > 0) * current_priority > saved_priority, current_priority, saved_priority).astype(np.uint8)

        return multilabel

    return None