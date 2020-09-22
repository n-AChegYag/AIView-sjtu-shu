import os
import argparse
import torch
import numpy as np
from scipy.ndimage import median_filter
import SimpleITK as sitk
from reg_sitk import *
from monai.transforms import Resize
import random


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=False, type=str)
    parser.add_argument("-m", "--mode", type=str, help='pixel or sample', default='pixel')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # args.output_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)

    model = torch.load('/workspace/mri_model.pth')['model']
    model = model.cuda()

    pred(model, args)


def pred(model, args):
    print("==> mode: {}".format(args.mode))

    to_transforms = Resize((64, 64, 64))
    from_transforms = torch.nn.Upsample((256, 256, 256), mode="trilinear", align_corners=False)

    registration_initialization()
    
    for f_name in os.listdir(args.input_dir):
        if not f_name.endswith('gz'):
            continue
        print('\nfile name:', f_name)
        
        # load image
        filepath = os.path.join(args.input_dir, f_name)
        mov = sitk.ReadImage(filepath, sitk.sitkFloat32)
        spacing = mov.GetSpacing()
        direction = mov.GetDirection()
        origin = mov.GetOrigin()
        
        # registration
        mov_info = get_image_information(mov)
        mov = image_preprocessing(mov, 'brain')
        aff_transform = brain_registration_execute(mov)
        mov_aff = fwd_transform(mov, aff_transform, sitk.sitkLinear)
        x = sitk.GetArrayFromImage(mov_aff)
        x = x.transpose(2,1,0)
        
        # restruction
        x = to_transforms(x[None])[None]
        x = torch.from_numpy(x).float()
        x = x.cuda()
        x = (x - 0.5) / 0.5
        model.eval()
        with torch.no_grad():
            x_r = model(x)
        x = (from_transforms(x*0.5+0.5))
        x_r = (from_transforms(x_r*0.5+0.5))
        
        # anomaly score 
        if args.mode == 'sample':
            sample_score = (x - x_r).abs().mean().cpu()
            sample_score = (sample_score - 0.01) / 0.008
            sample_score = torch.clamp(sample_score, 0, 1).item()
            print('====>sample score:', str(sample_score))
            with open(os.path.join(args.output_dir, f_name+".txt"), "w") as target_file:
                target_file.write(str(sample_score))
            
        elif args.mode == 'pixel':
            pixel_scores = torch.abs(x - x_r) / 2
            pixel_scores = median_filter(pixel_scores[0][0].cpu().numpy(), size=5)
            pixel_scores = pixel_scores.transpose(2,1,0)
            pixel_scores = sitk.GetImageFromArray(pixel_scores)
            pixel_scores = inv_transform(pixel_scores, aff_transform, sitk.sitkNearestNeighbor, 'brain', mov_info)
            pixel_scores.SetOrigin(origin)
            pixel_scores.SetSpacing(spacing)
            pixel_scores.SetDirection(direction)
            sitk.WriteImage(pixel_scores, os.path.join(args.output_dir, f_name))
            print('pixel score is saved as', os.path.join(args.output_dir, f_name))
       
    print('Finish\n')

    


if __name__ == "__main__":
	main()