import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd


def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def load_nii(directory):
    mask = nib.load(directory)
    mask = mask.get_fdata().transpose(2, 0, 1)

    return mask


def preproc():
    path_subset = '/subset/subset/'
    path_masks = '/subset/subset_masks/'
    path_list = []
    mask_path = []
    for i in os.listdir(path_subset):
        path_chosen_subset = path_subset + i + '/' + os.listdir(path_subset + i)[0] + \
                         '/' + os.listdir(path_subset + i + '/' + os.listdir(path_subset + i)[0])[0] + '/' + \
                         os.listdir(path_subset + i + '/' + os.listdir(path_subset + i)[0] + \
                                    '/' + os.listdir(path_subset + i + '/' + os.listdir(path_subset + i)[0])[0])[0]
        path_chosen_subset = path_chosen_subset.replace('.json', '')
        print(path_chosen_subset)
        saved_array = load_dicom(path_chosen_subset)

        for j in range(saved_array.shape[0]):
            saved_slice = saved_array[j, :, :]
            save_path = path_subset + i + f'/img{j}.png'
            Image.fromarray(saved_slice.astype('uint8')).save(save_path)
            path_list.append(save_path)

        path_chosen_mask = path_masks + i + '/' + os.listdir(path_masks + i)[0]
        saved_array = load_nii(path_chosen_mask).astype(np.int16)
        for j in range(saved_array.shape[0]):
            saved_slice = saved_array[j, :, :]
            save_path = path_masks + i + f'/img{j}.png'
            Image.fromarray(saved_slice.astype('uint8')).save(save_path)
            mask_path.append(save_path)

    df = pd.DataFrame({'img': path_list, 'mask': mask_path})
    df.to_csv('/subset/df.csv', index=False)
    return df
