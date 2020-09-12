import torch
from torch.autograd import Variable
import PIL

def read_gif_file(filename, seek_pos=1):
	img = PIL.Image.open(filename)
	try:
		img.seek(seek_pos)
	except EOFError:
		return img
	return img


class StoryGANSSIMDataset(torch.utils.data.Dataset):
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
        return fake[0], real_imgs

    def __len__(self):
        return self.dataset_size
