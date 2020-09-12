import torch
from torch.autograd import Variable

def onehot2embedding_idx(labels):
    labels_t = labels.T
    character = torch.zeros(labels.shape)
    for idx in range(labels.shape[1]):
        character[labels_t[idx, :] > 0, idx] = idx+1
    return character.long()


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index]['images']

    def __len__(self):
        return len(self.orig)


class GenerativeDataset(torch.utils.data.Dataset):
    """Only support DataLoader with num_worker=1 and image dataset"""

    def __init__(self, generator, z_dim, dataset_size, testdataset, device):
        self.generator = generator
        self.testdataset = testdataset
        self.z_dim = z_dim
        self.dataset_size = dataset_size
        self.device = device

    def __getitem__(self, index):        
        z = torch.randn(1, self.z_dim)
        real = self.testdataset[index]

        im_motion_input = real['description']
        label = torch.from_numpy(real['label']).unsqueeze(0)
        description = torch.from_numpy(real['description']).unsqueeze(0)
        embedding =  onehot2embedding_idx(label).to(self.device)
        z = torch.cat((z, description), axis=1).to(self.device)

        self.generator.eval()
        with torch.no_grad():
            image = self.generator(z, embedding)
        self.generator.train()
        return image[0]

    def __len__(self):
        return self.dataset_size

class StoryGANDataset(torch.utils.data.Dataset):
    """Only support DataLoader with num_worker=1 and image dataset"""

    def __init__(self, generator, dataset_size, testdataset):
        self.generator = generator
        self.testdataset = testdataset
        self.dataset_size = dataset_size

    def __getitem__(self, index):        
        real = self.testdataset[index]

        real_cpu = real['images']
        motion_input = real['description']
        content_input = real['description']
        real_imgs = Variable(real_cpu)
        motion_input = Variable(motion_input)
        content_input = Variable(content_input)
        labels = Variable(real['labels'])
        if next(self.generator.parameters()).is_cuda:
            real_imgs = real_imgs.cuda()
            labels = labels.cuda()
            motion_input = motion_input.cuda()
            content_input = content_input.cuda()
            motion_input = torch.cat((motion_input, labels), 1) 

        motion_input = motion_input.unsqueeze(0)
        content_input = content_input.unsqueeze(0)

        self.generator.eval()
        with torch.no_grad():
            _, fake, _,_,_,_,_ = self.generator.sample_videos(motion_input, content_input)
        self.generator.train()
        return fake[0]

    def __len__(self):
        return self.dataset_size