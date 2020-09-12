import os
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL

class StoryDataset(torch.utils.data.Dataset):

    def __init__(self, image_path, 
        transform, is_train = True):
        self.dir_path = image_path
        self.descriptions = np.load(image_path +'CLEVR_dict.npy', allow_pickle=True, encoding = 'latin1' ).item()
        self.transforms = transform

        self.srt = 0
        self.edn = 10000

        if not is_train:
            self.srt = 10000 # offset?
            self.edn = 13000

        self.video_len = 4

    def __getitem__(self, item):
        label = []
        super_label = []
        image = []
        des = []
        item = item + self.srt
        for i in range(self.video_len):
            v = '%simages/CLEVR_new_%06d_%d.png' % (self.dir_path, item, i+1)
            image_pos = v.split('/')[-1]
            im = np.array(PIL.Image.open(v))
            image.append( np.expand_dims(im[...,:3], axis = 0) )

            des.append(np.expand_dims(self.descriptions[image_pos].astype(np.float32), axis = 0))
            l = des[-1].reshape(-1)
            label.append(l[i*18 + 3: i*18 + 11])
            super_label.append(l[i*18:i*18+15])

        label[0] = np.expand_dims(label[0], axis = 0)
        super_label[0] = np.expand_dims(super_label[0], axis = 0)
        for i in range(1,4):
            label[i] = label[i] + label[i-1]
            super_label[i] = super_label[i] + super_label[i-1]
            temp = label[i].reshape(-1)
            super_temp = super_label[i].reshape(-1)
            temp[temp>1] = 1
            super_temp[super_temp>1] = 1
            label[i] = np.expand_dims(temp, axis = 0)
            super_label[i] = np.expand_dims(super_temp, axis = 0)
        des = np.concatenate(des, axis = 0)
        image_numpy = np.concatenate(image, axis = 0)
        image = self.transforms(image_numpy)
        label = np.concatenate(label, axis = 0)
        super_label = np.concatenate(super_label, axis = 0)
        # image is T x H x W x C
        # After transform, image is C x T x H x W
        des = torch.tensor(des)
        ## des is attribute, subs is encoded text description
        return {'images': image,
                'description': des,
            'labels': super_label}

    def __len__(self):
        return self.edn - self.srt + 1


class ImageDataset(torch.utils.data.Dataset):
    # make sure hyperparameters are same as pororoSV
    def __init__(self, image_path, transform, 
        segment_transform=None, use_segment=False, segment_name='img_segment',
        is_train = True):
        self.dir_path = image_path
        self.transforms = transform
        self.segment_transform = segment_transform
        self.descriptions = np.load(image_path +'CLEVR_dict.npy', allow_pickle=True, encoding = 'latin1').item()
        self.transforms = transform
        self.use_segment = use_segment

        self.srt = 0
        self.edn = 10000

        if not is_train:
            self.srt = 10000 # offset?
            self.edn = 13000

        self.video_len = 4

    def __getitem__(self, item):
        item = item + self.srt
        se = np.random.randint(1,self.video_len+1, 1)
        path = '%simages/CLEVR_new_%06d_%d.png' % (self.dir_path, item, se)

        im = PIL.Image.open(path)
        image = np.array(im)[...,:3]
        image = self.transforms(image)
        img_pos = path.split('/')[-1]

        des = self.descriptions[img_pos].astype(np.float32)
        label = des[3:11]
        super_label = des[:15]
        content = []
        for i in range(self.video_len):
            v = '%simages/CLEVR_new_%06d_%d.png' % (self.dir_path, item, i+1)
            img_pos = v.split('/')[-1]
            content.append(np.expand_dims(self.descriptions[img_pos].astype(np.float32), axis = 0))

        for i in range(1,4):
            label = label + des[i*18 + 3: i*18 + 11]
            super_label = super_label + des[i*18:i*18+15]
        label = label.reshape(-1)
        super_label = super_label.reshape(-1)
        label[label>1] = 1
        super_label[super_label>1] = 1
        content = np.concatenate(content, 0)
        content = torch.tensor(content)
        ## des is attribute, subs is encoded text description
        output =  {'images': image,
                'description': des, 
                'labels':super_label, 
                'content': content 
            }

        # load segment image label
        if self.use_segment:
            mask_name = '%simages/CLEVR_new_%06d_%d_mask.png' % (self.dir_path, item, i+1)
            mask_im = PIL.Image.open(mask_name).convert('L')
            mask_im = self.segment_transform( np.array(mask_im) )
            output['images_seg'] = mask_im
        return output

    def __len__(self):
        return self.edn - self.srt + 1



if __name__ == "__main__":
    from datasets.utils import video_transform
    n_channels = 3
    image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            lambda x: x[:n_channels, ::],
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
    image_transforms_seg = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize((64, 64) ),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    video_transforms = functools.partial(video_transform, image_transform=image_transforms)
    imagedataset = ImageDataset('./CLEVR/', (image_transforms, image_transforms_seg))
    storydataset = StoryDataset('./CLEVR/', video_transforms)
    storyloader = torch.utils.data.DataLoader(
            imagedataset, batch_size=13,
            drop_last=True, shuffle=True, num_workers=8)
    for batch in storyloader:
        print(batch['description'].shape, batch['content'].shape, batch['labels'].shape)
        break