from PIL import Image
import os
import os.path
import time
import pdb
import pandas as pd
import numpy as np

def extractFrames(inGif, outFolder, img_name):
	frame = Image.open(inGif)
	#nframes = 0 
    #while frame:
	## Save the first frame
	for i in range(1):
		#frame.save( '%s/%s-%s.gif' % (outFolder, os.path.basename(inGif), nframes ) , 'GIF')
		frame.save('{}/{}.png'.format(outFolder, img_name), 'png')
		#nframes += 1
		#try:
		#	frame.seek( nframes )
		#except EOFError:
		#	break;
	return True


def get_eps_sentences(descriptions, ep, max_num):
	temp = 1
	num_skip = 0
	sentences = []
	str_index = descriptions[descriptions[0]==ep].index[0]
	for i in range(str_index, len(descriptions)):
		order = descriptions.loc[i][1]
		if order == max_num:
			sentence = descriptions.loc[i][2]
			sentences.append(sentence)
			#print("temp:",temp)
			return sentences
		
		elif order == temp:
			## Get the first appear sentence
			sentence = descriptions.loc[i][2]
			sentences.append(sentence)
			#print("temp:", temp)
			temp += 1
		else:
			#print("Skip the repeated sentence")
			num_skip += 1

def get_encoded_vec(des_vec, E, ep, num_gif):
	values = []
	for i in range(num_gif):
		key = '{}/{}/{}'.format(E, ep, i+1)
		try:
			value = des_vec[str.encode(key)][0]
		except:
			print("Error in description vec")
			pdb.set_trace()
		values.append(value)
	return values

def obtain_pororo_dict():
	descriptions = pd.read_csv("descriptions.csv", header=None)
	descriptions_vec = np.load("pororo_png/descriptions_vec.npy", encoding='bytes').item()

	Es = ["Pororo_ENGLISH1_1","Pororo_ENGLISH1_2","Pororo_ENGLISH1_3","Pororo_ENGLISH2_1","Pororo_ENGLISH2_2","Pororo_ENGLISH2_3","Pororo_ENGLISH2_4","Pororo_ENGLISH3_1","Pororo_ENGLISH3_2","Pororo_ENGLISH3_3","Pororo_ENGLISH3_4","Pororo_ENGLISH4_1","Pororo_ENGLISH4_2", "Pororo_ENGLISH4_3","Pororo_ENGLISH4_4", "Pororo_Rescue", "Pororo_The_Racing_Adventure"]
	output_file = "Pororo_dataset_new"
	num_story = 0
	ckt_process = 1
	num_png = 0
	info_dict = dict() # key=img_name, value=description
	now_time = time.time()

	for E in Es:
		path_E = "Pororo_dataset/Scenes_Dialogues/" + E
		eps = os.listdir(path_E)

		for ep in eps:
			path_E_ep = os.path.join(path_E, ep)
			gif_files = os.listdir(path_E_ep)

			num_gif = len(gif_files)-1 # remove non gif file
			#sentences = get_eps_sentences(descriptions, ep, num_gif)
			encoded_vecs = get_encoded_vec(descriptions_vec, E, ep, num_gif)
		
			for i in range(num_gif-4):
				for j in range(5):
					# Get an image 
					gif_file = '{}.gif'.format(str(i+1+j))
					path_gif = os.path.join(path_E_ep, gif_file)
					img_name = ep + "_" + "{:06}".format(num_story) + "_" + str(j+1)
					#print("img_name", img_name)
					extractFrames(path_gif, output_file, img_name)
					num_png += 1

					# Format the information
					#info_dict[img_name]=sentences[i+j]
					info_dict[str.encode(img_name+'.png')]=encoded_vecs[i+j]
				num_story += 1
		print("{}/{} is done in {}".format(ckt_process, len(Es), time.time()-now_time))
		ckt_process += 1

	print("The total number of picture: {}".format(num_png))
	np.save("Pororo_dict.npy", info_dict)


if __name__ == "__main__":
	import os, glob, PIL
	for filename in glob.glob('/mnt/storage/pororoSV/Scenes_Dialogues/*/*/*.gif'):
		img_name = os.path.basename(filename)
		print(img_name)