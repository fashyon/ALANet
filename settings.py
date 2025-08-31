from os.path import join

# experiment specifics

name = 'ALANet' #'name of the experiment. It decides where to store samples and models'
gpu_ids = '0' #'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
checkpoints_dir = './checkpoints' #'models are saved here'
resume_epoch = None #'checkpoint to use. (default: latest')
seed = 2018 #'random seed to use. Default=2018'
isTrain = True

# modify the following code to
datadir = '../data/mytrain/train'
datadir_test = '../data/mytrain/test'
datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real')
datadir_nature = join(datadir, 'nature')
datadir_flickr8k = "../data/flickr8k"


# for training
lr = 1e-4 #initial learning rate for adam
wd = 0 #weight decay for adam
low_sigma = 2 #min sigma in synthetic dataset
high_sigma = 5 #max sigma in synthetic dataset
low_gamma = 1.3 #max gamma in synthetic dataset
high_gamma = 1.3 #max gamma in synthetic dataset

# data augmentation
batchSize = 1 #input batch size

# for setting input
serial_batches = False #'if true, takes images in order to make batches, otherwise takes them randomly'
nThreads = 8 #'# threads for loading data'
max_dataset_size = None #'Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.'


# loss weight
lambda_mse = 1.0
lambda_grad = 2.0
lambda_vgg = 0.01


# for displays
no_html = False #do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/
save_epoch_freq = 5 #frequency of saving checkpoints at the end of epochs
# for display
no_log = False #'disable tf logger?'
display_winsize = 256 #'display window size'
display_port = 8097 #'visdom port of the web display'
display_id = 0 #'window id of the web display (use 0 to disable visdom)'
display_single_pane_ncols = 0 #'if positive, display all images in a single visdom web panel with certain number of images per row.'

