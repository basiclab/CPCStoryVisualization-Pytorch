import os
from os.path import exists
import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import re

class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, counter = None, cache=None, min_len=4, data_type='train'):
        assert data_type in ['train', 'test', 'valid']
        self.lengths = []
        self.followings = []
        dataset = ImageFolder(folder)
        self.dir_path = folder
        self.total_frames = 0
        self.images = []
        self.labels = np.load(os.path.join(folder,'labels.npy'),
            allow_pickle=True, encoding = 'latin1').item()
        path_img_cache = os.path.join(cache, "img_cache{}.npy".format(min_len))
        path_follow_cache = os.path.join(cache, "following_cache{}.npy".format(min_len))

        if cache is not None and exists(path_img_cache) and exists(path_follow_cache) :
            self.images     = np.load( path_img_cache,    allow_pickle=True, encoding = 'latin1')
            self.followings = np.load( path_follow_cache, allow_pickle=True, encoding = 'latin1')
        else:
            for idx, (im, _) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                v_name = img_path.replace(folder,'') # get the video name
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(folder, ''))
                self.followings.append(following_imgs)
            np.save(folder + 'img_cache' + str(min_len) + '.npy', self.images)
            np.save(folder + 'following_cache' + str(min_len) + '.npy', self.followings)
        train_id, test_id = np.load(self.dir_path + 'train_test_ids.npy', allow_pickle=True, encoding = 'latin1')
        orders = train_id if data_type == 'train' else test_id
        orders = np.array(orders).astype('int32')
        self.images = self.images[orders]
        self.followings = self.followings[orders]
        print("[{}] Total number of clips {}".format(data_type,len(self.images)))


    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = longer // shorter
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):
        # return a training list
        lists = [self.images[item]]
        for i in range(len(self.followings[item])):
            lists.append(str(self.followings[item][i]))
        return lists

    def __len__(self):
        return len(self.images)

class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, textvec, transform):
        # dataset: VideoFolderDataset
        # textvec: dir_path, a path to the npy file about the text
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        lat = 'latin1'
        self.descriptions = np.load(os.path.join(textvec, 'descriptions_vec.npy'),
            allow_pickle=True,encoding=lat).item()
        self.attributes = np.load(os.path.join(textvec, 'descriptions_attr.npy'),
            allow_pickle=True,encoding=lat).item()
        self.subtitles = np.load(os.path.join(textvec, 'subtitles_vec.npy'),
            allow_pickle=True,encoding=lat).item()
        self.descriptions_original = np.load(os.path.join(textvec,'descriptions.npy'),
            allow_pickle=True,encoding=lat).item()

        self.transforms = transform
        self.labels = dataset.labels # character labels

    def save_story(self, output, save_path = './'):
        all_image = []
        images = output['images_numpy']
        texts = output['text']
        for i in range(images.shape[0]):
            all_image.append(np.squeeze(images[i]))
        output = PIL.Image.fromarray(np.concatenate(all_image, axis = 0))
        output.save(save_path + 'image.png')
        fid = open(save_path + 'text.txt', 'w')
        for i in range(len(texts)):
            fid.write(texts[i] +'\n' )
        fid.close()
        return

    def __getitem__(self, item):
        lists = self.dataset[item] # list that contains a sequence of image names
        labels = []
        image = []
        subs = []
        des = []
        attri = []
        text = []
        for v in lists:

            if type(v) == np.bytes_:
                v = v.decode('utf-8')
            elif v.split('\'')[0] == 'b':
                v = v.replace('b', '').replace('\'', '')

            id = v.replace('.png', '')
            path = self.dir_path + id + '.png'
            im = PIL.Image.open(path) # open the image
            im = im.convert('RGB')    # convert to RGB
            image.append( np.expand_dims(np.array( self.dataset.sample_image(im)), axis = 0) ) # cropping
            se = 0
            if len(self.descriptions_original[id]) > 1:
                # if the description for an image is more than 1, than random pick one
                se = np.random.randint(0,len(self.descriptions_original[id]),1)
                se = se[0]
            text.append(  self.descriptions_original[id][se])
            des.append(np.expand_dims(self.descriptions[id][se], axis = 0))
            subs.append(np.expand_dims(self.subtitles[id][0], axis = 0))
            labels.append(np.expand_dims(self.labels[id], axis = 0))
            attri.append(np.expand_dims(self.attributes[id][se].astype('float32'), axis = 0))
        subs = np.concatenate(subs, axis = 0)
        attri = np.concatenate(attri, axis = 0)
        des = np.concatenate(des, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        image_numpy = np.concatenate(image, axis = 0)
        # image is T x H x W x C
        image = self.transforms(image_numpy)  # permuation and convert numpy into torch
        # After transform, image is C x T x H x W
        ##
        des = np.concatenate([des, attri], 1) # shape: 5, 128+228=356
        ##

        des = torch.tensor(des)
        subs = torch.tensor(subs)
        attri = torch.tensor(attri)
        labels = torch.tensor(labels.astype(np.float32))

        return {'images': image, 'text':text, 'description': des,
                'subtitle': subs, 'images_numpy':image_numpy, 'labels':labels}

    def __len__(self):
        return len(self.dataset.images)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, textvec, image_transform, 
        segment_transform=None, use_segment=False, segment_name='img_segment'):
        # dataset: VideoFolderDataset
        # tectvec: a path to the npy file about text
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.use_segment = use_segment
        lat = 'latin1'
        self.descriptions = np.load(os.path.join(textvec,'descriptions_vec.npy'),
            allow_pickle=True,encoding=lat).item()
        self.attributes =  np.load(os.path.join(textvec,'descriptions_attr.npy'),
            allow_pickle=True,encoding=lat).item()
        self.subtitles = np.load(os.path.join(textvec,'subtitles_vec.npy'),
            allow_pickle=True,encoding=lat).item()
        self.descriptions_original = np.load(os.path.join(textvec,'descriptions.npy'),
            allow_pickle=True,encoding=lat).item()
        self.transforms = image_transform
        self.transforms_seg = segment_transform
        self.labels = dataset.labels
        self.segment_name = segment_name
        print('segment dir: ', self.segment_name)

    def __getitem__(self, item):
        # Read segmentation image
        sub_path = self.dataset[item][0].decode('utf-8')
        if self.use_segment:
            path = '{}/{}/{}'.format(self.dir_path, self.segment_name, '_'.join(sub_path.split('/')[-2:]) )#self.dir_path+'/''/'+'_'.join(sub_path.split('/')[-2:])
            im = PIL.Image.open(path)
            # v2
            im = im.convert('L')
            image_seg = np.array( self.dataset.sample_image(im))
            image_seg = self.transforms_seg(image_seg)

        # Read orginal image
        path = self.dir_path + sub_path
        im = PIL.Image.open(path)
        im = im.convert('RGB')
        image = np.array( self.dataset.sample_image(im))
        image = self.transforms(image)

        # Read subtitle
        id = sub_path.replace('.png','')
        subs = self.subtitles[id][0]

        # Read text des (embedding),text attribute (one-hot), raw text, character label (one-hot)
        se = 0
        if len(self.descriptions_original[id]) > 1:
            se = np.random.randint(0,len(self.descriptions_original[id]),1)
            se = se[0]
        des = self.descriptions[id][se]
        attri = self.attributes[id][se].astype('float32')
        text = self.descriptions_original[id][se]
        label = self.labels[id].astype(np.float32)

        lists = self.dataset[item]
        content = []
        attri_content = []
        attri_label = []
        for v in lists:
            if type(v) == np.bytes_:
                v = v.decode('utf-8')
            elif v.split('\'')[0] == 'b':
                v = v.replace('b', '').replace('\'', '')

            img_id = v.replace('.png','')
            se = 0
            if len(self.descriptions[img_id]) > 1:
                se = np.random.randint(0,len(self.descriptions[img_id]),1)
                se = se[0]
            content.append(np.expand_dims(self.descriptions[img_id][se], axis = 0))
            attri_content.append(np.expand_dims(self.attributes[img_id][se].astype('float32'), axis = 0))
            attri_label.append(np.expand_dims(self.labels[img_id].astype('float32'), axis = 0))

        base_content = np.concatenate(content, axis = 0)
        attri_content = np.concatenate(attri_content, axis = 0)
        attri_label = np.concatenate(attri_label, axis = 0)
        content = np.concatenate([base_content, attri_content, attri_label], 1)

        des = np.concatenate([des, attri])
        ##
        content = torch.tensor(content)
        output = {'images': image, 'text':text, 'description': des,
                'subtitle': subs, 'labels':label, 'content': content }

        if self.use_segment:
            output['images_seg'] = image_seg

        return output

    def __len__(self):
        return len(self.dataset.images)


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

    # test run here
    counter = np.load('/mnt/storage/img_pororo/frames_counter.npy', allow_pickle=True).item()
    base= VideoFolderDataset('/mnt/storage/img_pororo/', counter = counter, cache = '/mnt/storage/img_pororo/')
    imagedataset = ImageDataset(base, '/mnt/storage/img_pororo/', (image_transforms, image_transforms_seg))
    storydataset = StoryDataset(base, '/mnt/storage/img_pororo/', video_transforms)
    storyloader = torch.utils.data.DataLoader(
            imagedataset, batch_size=13,
            drop_last=True, shuffle=True, num_workers=8)
    for batch in storyloader:
        print(batch['description'].shape)