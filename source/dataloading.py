from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

class KiTSDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        image = Image.open(img_path)
        segmentation = Image.open(seg_path)

        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)

        return image, segmentation



class CroppedKiTSDataset(Dataset):
    def __init__(self, base_path, transform=None, crop_size=None):
        self.transform = transform
        self.image_paths = []
        self.crop_size = crop_size
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)
    
    def center_crop(self, image, mask):
        """
        Center crop both the image and mask to the specified size.
        The image is a PIL Image and the mask is a NumPy array.
        """
        # Image dimensions
        image_width, image_height = image.size
        mask_height, mask_width = mask.shape[1:]  # Correct unpacking here

        crop_width, crop_height = self.crop_size

        # Calculate coordinates for center crop
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Crop the mask (assumed to be in CHW format)
        mask = mask[:, top:bottom, left:right]

        return image, mask
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        # Load the segmentation mask as a NumPy array
        segmentation = np.array(Image.open(seg_path))

        # Handle 3-channel segmentation masks
        if segmentation.ndim == 3 and segmentation.shape[-1] == 3:
            segmentation = np.transpose(segmentation, (2, 0, 1))  # Convert to CHW format
        elif segmentation.ndim == 2:  # If single channel, expand to CHW
            segmentation = np.expand_dims(segmentation, axis=0)  # Add channel dimension

        # Apply center crop if crop_size is specified
        if self.crop_size:
            image, mask = self.center_crop(image, segmentation)

        # Apply transforms to the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Convert mask to a torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

class CroppedKiTSDataset4(Dataset):
    def __init__(self, base_path, transform=None, crop_size=None):
        self.transform = transform
        self.image_paths = []
        self.crop_size = crop_size
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)
    

    def center_crop(self, image, kidney_mask):
        """
        Center crop both the image and kidney mask to the specified size.
        The image is a PIL Image and the kidney_mask is a NumPy array.
        """
        # Image dimensions
        image_width, image_height = image.size
        mask_height, mask_width = kidney_mask.shape[1:]

        crop_width, crop_height = self.crop_size

        # Calculate coordinates for center crop
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Crop the kidney mask (which is a NumPy array)
        kidney_mask = kidney_mask[:, top:bottom, left:right]

        return image, kidney_mask
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        # Load the segmentation mask as a NumPy array
        segmentation = np.array(Image.open(seg_path))

        # Handle 4-channel segmentation masks
        if segmentation.shape[-1] == 4:
            segmentation = np.transpose(segmentation, (2, 0, 1))


        # Apply center crop if crop_size is specified
        if self.crop_size:
            image, mask = self.center_crop(image, segmentation)

        # Apply transforms to the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Convert kidney_mask to a torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask



class KidneyOnly(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        segmentation = np.array(Image.open(seg_path))

        if segmentation.shape[-1] == 4:  
            segmentation = np.transpose(segmentation, (2, 0, 1))  

        if segmentation.ndim == 3 and segmentation.shape[0] == 4:
            kidney_mask = segmentation[1:2, :, :]  
        else:
            raise ValueError("Unexpected segmentation shape, expected (4, height, width)")

        if self.transform:
            image = self.transform(image)

        kidney_mask = torch.tensor(kidney_mask, dtype=torch.float32)

        return image, kidney_mask


class CroppedKidneyOnly(Dataset):
    def __init__(self, base_path, transform=None, crop_size=None):
        self.transform = transform
        self.image_paths = []
        self.crop_size = crop_size
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)
    

    def center_crop(self, image, kidney_mask):
        """
        Center crop both the image and kidney mask to the specified size.
        The image is a PIL Image and the kidney_mask is a NumPy array.
        """
        # Image dimensions
        image_width, image_height = image.size
        mask_height, mask_width = kidney_mask.shape[1:]

        crop_width, crop_height = self.crop_size

        # Calculate coordinates for center crop
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Crop the kidney mask (which is a NumPy array)
        kidney_mask = kidney_mask[:, top:bottom, left:right]

        return image, kidney_mask
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        # Load the segmentation mask as a NumPy array
        segmentation = np.array(Image.open(seg_path))

        # Handle 4-channel segmentation masks
        if segmentation.shape[-1] == 4:
            segmentation = np.transpose(segmentation, (2, 0, 1))

        # Extract kidney mask (class 1)
        if segmentation.ndim == 3 and segmentation.shape[0] == 4:
            kidney_mask = segmentation[1:2, :, :]
        else:
            raise ValueError("Unexpected segmentation shape, expected (4, height, width)")

        # Apply center crop if crop_size is specified
        if self.crop_size:
            image, kidney_mask = self.center_crop(image, kidney_mask)

        # Apply transforms to the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Convert kidney_mask to a torch tensor
        kidney_mask = torch.tensor(kidney_mask, dtype=torch.float32)

        return image, kidney_mask



class CroppedTumorOnly(Dataset):
    def __init__(self, base_path, transform=None, crop_size=None):
        self.transform = transform
        self.image_paths = []
        self.crop_size = crop_size
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)
    

    def center_crop(self, image, tumor_mask):

        # Image dimensions
        image_width, image_height = image.size
        mask_height, mask_width = tumor_mask.shape[1:]

        crop_width, crop_height = self.crop_size

        # Calculate coordinates for center crop
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Crop the kidney mask (which is a NumPy array)
        tumor_mask = tumor_mask[:, top:bottom, left:right]

        return image, tumor_mask
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        # Load the segmentation mask as a NumPy array
        segmentation = np.array(Image.open(seg_path))

        # Handle 4-channel segmentation masks
        if segmentation.shape[-1] == 4:
            segmentation = np.transpose(segmentation, (2, 0, 1))

        # Extract kidney mask (class 1)
        if segmentation.ndim == 3 and segmentation.shape[0] == 4:
            tumor_mask = segmentation[2:3, :, :]
        else:
            raise ValueError("Unexpected segmentation shape, expected (4, height, width)")

        # Apply center crop if crop_size is specified
        if self.crop_size:
            image, tumor_mask = self.center_crop(image, tumor_mask)

        # Apply transforms to the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Convert kidney_mask to a torch tensor
        tumor_mask = torch.tensor(tumor_mask, dtype=torch.float32)

        return image, tumor_mask

class CroppedKidneyAndTumor(Dataset):
    def __init__(self, base_path, transform=None, crop_size=None):
        self.transform = transform
        self.image_paths = []
        self.crop_size = crop_size
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)
    

    def center_crop(self, image, kidney_mask):
        """
        Center crop both the image and kidney mask to the specified size.
        The image is a PIL Image and the kidney_mask is a NumPy array.
        """
        # Image dimensions
        image_width, image_height = image.size
        mask_height, mask_width = kidney_mask.shape[1:]

        crop_width, crop_height = self.crop_size

        # Calculate coordinates for center crop
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Crop the kidney mask (which is a NumPy array)
        kidney_mask = kidney_mask[:, top:bottom, left:right]

        return image, kidney_mask
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        # Load the segmentation mask as a NumPy array
        segmentation = np.array(Image.open(seg_path))

        # Handle 4-channel segmentation masks
        if segmentation.shape[-1] == 4:
            segmentation = np.transpose(segmentation, (2, 0, 1))

        # Extract kidney mask (class 1)
        if segmentation.ndim == 3 and segmentation.shape[0] == 4:
            mask = segmentation[1:3, :, :]
        else:
            raise ValueError("Unexpected segmentation shape, expected (4, height, width)")

        # Apply center crop if crop_size is specified
        if self.crop_size:
            image, mask = self.center_crop(image, mask)

        # Apply transforms to the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Convert kidney_mask to a torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask
    
    

class CroppedwoBackground(Dataset):
    def __init__(self, base_path, transform=None, crop_size=None):
        self.transform = transform
        self.image_paths = []
        self.crop_size = crop_size
        self.segmentation_paths = []

        # Iterate over all case directories
        for case_dir in os.listdir(base_path):
            imaging_dir = os.path.join(base_path, case_dir, 'imaging')
            segmentation_dir = os.path.join(base_path, case_dir, 'segmentation')

            if os.path.isdir(imaging_dir) and os.path.isdir(segmentation_dir):
                imaging_files = sorted(os.listdir(imaging_dir))
                segmentation_files = sorted(os.listdir(segmentation_dir))

                # Store full paths to the images and segmentations
                for img_file, seg_file in zip(imaging_files, segmentation_files):
                    img_path = os.path.join(imaging_dir, img_file)
                    seg_path = os.path.join(segmentation_dir, seg_file)

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)
    

    def center_crop(self, image, kidney_mask):
        """
        Center crop both the image and kidney mask to the specified size.
        The image is a PIL Image and the kidney_mask is a NumPy array.
        """
        # Image dimensions
        image_width, image_height = image.size
        mask_height, mask_width = kidney_mask.shape[1:]

        crop_width, crop_height = self.crop_size

        # Calculate coordinates for center crop
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Crop the kidney mask (which is a NumPy array)
        kidney_mask = kidney_mask[:, top:bottom, left:right]

        return image, kidney_mask
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        # Load the image
        image = Image.open(img_path)

        # Load the segmentation mask as a NumPy array
        segmentation = np.array(Image.open(seg_path))

        # Handle 4-channel segmentation masks
        if segmentation.shape[-1] == 4:
            segmentation = np.transpose(segmentation, (2, 0, 1))

        # Extract kidney mask (class 1)
        if segmentation.ndim == 3 and segmentation.shape[0] == 4:
            mask = segmentation[1:4, :, :]
        else:
            raise ValueError("Unexpected segmentation shape, expected (4, height, width)")

        # Apply center crop if crop_size is specified
        if self.crop_size:
            image, mask = self.center_crop(image, mask)

        # Apply transforms to the image (if specified)
        if self.transform:
            image = self.transform(image)

        # Convert kidney_mask to a torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask
    
    
    


