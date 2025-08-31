import os
from os.path import join
import model
import settings as settings
os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_ids
import util.util as util
import util.index as index
from util.visualizer import Visualizer
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import random
from PIL import Image
from collections import OrderedDict
import losses as losses
import gradio as gr


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(settings.seed)
np.random.seed(settings.seed)  # seed for every module
random.seed(settings.seed)

"""Main Loop"""
def tensor2im(image_tensor):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy

class Model():
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = os.path.join(settings.checkpoints_dir, settings.name)
        self._count = 0
        self.Tensor = torch.cuda.FloatTensor
        self.isTrain = True
        self.vgg = None

        if torch.cuda.device_count() == 1:
            print('one GPU runing!')
            self.vgg = model.Vgg19(requires_grad=False).to(self.device)
            self.network = model.ALANet().to(self.device)
        if torch.cuda.device_count() > 1:
            print('DataParallel GPU runing!')
            self.vgg = model.Vgg19(requires_grad=False).to(self.device)
            self.network = torch.nn.DataParallel(model.ALANet()).to(self.device)

        if self.isTrain:
            # define loss functions
            self.init_loss()
            self.vgg_loss = losses.VGGLoss(self.vgg)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.network.parameters(),
                                                lr=1e-4, betas=(0.9, 0.999), weight_decay=settings.wd)

            self._init_optimizer([self.optimizer_G])

        self.load(settings.resume_epoch)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 32
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def forward(self):

        input = self.input
        input_check,ori_size = self.check_image_size(input)
        feature_vgg = self.vgg(input_check) #[H,H/2,H/4,H/8,H/16] [64,128,256,512,512]
        output_t, output_r = self.network(input,self.caption_t,self.caption_r,feature_vgg)
        self.output_t = output_t
        self.output_r = output_r

        return output_t, output_r

    def backward_G(self):
        self.loss_G = 0
        self.loss_T_pixel = settings.lambda_mse * self.l2_loss(self.output_t, self.target_t) + settings.lambda_grad * self.gradient_loss(self.output_t, self.target_t)
        if self.target_r != None:
            self.loss_R_pixel =  settings.lambda_mse * self.l2_loss(self.output_r, self.target_r) + settings.lambda_grad * self.gradient_loss(
                self.output_r, self.target_r)
        else:
            self.loss_R_pixel = 0
        self.loss_T_vgg = self.vgg_loss(self.output_t, self.target_t) * settings.lambda_vgg
        self.loss_G = self.loss_T_pixel + self.loss_R_pixel + self.loss_T_vgg

        self.loss_G.backward()

    def init_loss(self):
        self.l2_loss = torch.nn.MSELoss()
        self.gradient_loss = losses.GradientLoss()

    def set_input(self, data, mode='train'):
        target_t = None
        data_name = None
        mode = mode.lower()
        caption_t = data.get('caption_t', None)
        caption_r = data.get('caption_r', None)
        target_r = data.get('target_r', None)
        if mode == 'train':
            input, target_t = data['input'], data['target_t']
        elif mode == 'eval':
            input, target_t, data_name = data['input'], data['target_t'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        input = input.to(device)  # (device=self.gpu_ids[0])
        if target_t is not None:
            target_t = target_t.to(device)  # to(device=self.gpu_ids[0])
        if target_r is not None:
            target_r = target_r.to(device)  # to(device=self.gpu_ids[0])

        self.input = input
        self.target_t = target_t
        self.data_name = data_name
        self.target_r = target_r
        self.caption_t = caption_t
        self.caption_r = caption_r

        self.aligned = False if 'unaligned' in data else True


    def eval(self, data, savedir=None):
        self.network.eval()
        self.set_input(data, 'eval')
        with torch.no_grad():
            self.forward()
            output_t = tensor2im(self.output_t)
            target_t = tensor2im(self.target_t)
            if self.aligned:
                res_T = index.quality_assess(output_t, target_t)
            else:
                res_T = {}
            if savedir is not None:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                name, file_extension = os.path.splitext(os.path.basename(self.data_name[0]))
                Image.fromarray(output_t.astype(np.uint8)).save(join(savedir,'{}_T_ssim{:.6f}_psnr{:.6f}.png'.format(
                                                                         name, res_T['SSIM'], res_T['PSNR'])))
            return res_T

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.save_dir, 'model_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.save_dir, 'model_' + label + '.pt')

        torch.save(self.state_dict(), model_name)

    def _init_optimizer(self, optimizers):
        for optimizer in optimizers:
            util.set_opt_param(optimizer, 'initial_lr', settings.lr)
            util.set_opt_param(optimizer, 'weight_decay', settings.wd)

    def optimize_parameters(self):
        self.network.train()
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss'] = self.loss_G.item()
        return ret_errors

    def load(self, resume_epoch=None):
        try:
            model_path = util.get_model_list(self.save_dir, 'model', epoch=resume_epoch)
            state_dict = torch.load(model_path)
        except FileNotFoundError:
            print('No checkpoint net%s!!' % model_path)
            return
        self.epoch = state_dict['epoch']
        self.iterations = state_dict['iterations']
        self.network.load_state_dict(state_dict['network'])
        if self.isTrain:
            self.optimizer_G.load_state_dict(state_dict['opt'])
        print('Model loaded successfully: ' + str(model_path))
        print('Resume from epoch %d, iteration %d' % (self.epoch, self.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'network': self.network.state_dict(),
            'opt': self.optimizer_G.state_dict(),
            'epoch': self.epoch,
            'iterations': self.iterations,
        }
        return state_dict

class Engine(object):
    def __init__(self):
        self.best_val_loss = 1e6
        self.visualizer = None
        self.PSNR_AvgdataTotal = 0
        self.SSIM_AvgdataTotal = 0
        self.Num_EvalDataset = 0
        self.__setup()

    def __setup(self):
        self.basedir = join(settings.checkpoints_dir, settings.name)
        print('self.basedir--------------------', self.basedir)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        """Model"""
        self.model = Model()

        if not settings.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer()

    # === Convenient single-image inference function (for Gradio use) ===
    @torch.no_grad()
    def predict_single(self, pil_input, caption_t=None, caption_r=None, save_dir=None, save_basename="single"):
        self.model.network.eval()
        # PIL -> Tensor [1,3,H,W], [0,1]
        img = pil_input.convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(device)

        data = {
            'input': ten,
            'target_t': None,
            'fn': [f'{save_basename}.png'],
            'caption_t': caption_t,
            'caption_r': caption_r,
        }
        self.model.set_input(data, mode='eval')
        out_t, out_r = self.model.forward()
        out_t_np = tensor2im(out_t).astype(np.uint8)
        out_r_np = tensor2im(out_r).astype(np.uint8)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(out_t_np).save(join(save_dir, f'{save_basename}_T.png'))
            Image.fromarray(out_r_np).save(join(save_dir, f'{save_basename}_R.png'))

        return Image.fromarray(out_t_np), Image.fromarray(out_r_np)


    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, **kwargs):

        avg_meters = util.AverageMeters()
        model = self.model
        with torch.no_grad():
            PSNR_total = 0.0
            SSIM_total = 0.0
            for i, data in enumerate(val_loader):
                index = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index)
                PSNR_total += index['PSNR']
                SSIM_total += index['SSIM']
                util.progress_bar(i, len(val_loader), str(avg_meters))
        average_PSNR_value = round(PSNR_total / len(val_loader),6)
        average_SSIM_value = round(SSIM_total / len(val_loader),6)
        print('average PSNR {} ,average SSIM {}, on {} test_imgs ({})'.format(average_PSNR_value, average_SSIM_value,len(val_loader),
                                                                  dataset_name))
        self.PSNR_AvgdataTotal = self.PSNR_AvgdataTotal + average_PSNR_value
        self.SSIM_AvgdataTotal = self.SSIM_AvgdataTotal + average_SSIM_value

        logfile = open(self.basedir + '/loss_log.txt','a+')
        logfile.write('step  = ' + str(self.iterations) + '\t'
                      'epoch = ' + str(self.epoch) + '\t'
                      'lr = ' + str(model.optimizer_G.param_groups[0]['lr']) + '\t'
                      'average PSNR {} ,average SSIM {}, on {} test_imgs ({})'.format(average_PSNR_value, average_SSIM_value,len(val_loader),dataset_name)+ '\t''\n')
        logfile.close()
        self.Num_EvalDataset = self.Num_EvalDataset+1

        if not settings.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)

        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' %
                      (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e

engine = Engine()


# ====== Gradio Callback ======
def ui_single_infer(input_img, caption_t, caption_r, save_results, save_subdir):
    if input_img is None:
        return None, None, "Please upload an input image."
    save_dir = None
    if save_results:
        base_dir = './results_eval'
        save_dir = join(base_dir, 'single_infer' if not save_subdir else save_subdir)
    outT, outR = engine.predict_single(
        pil_input=input_img,
        caption_t=caption_t if caption_t else None,
        caption_r=caption_r if caption_r else None,
        save_dir=save_dir,
        save_basename="input"
    )
    msg = "Done."
    if save_dir:
        msg += f" Results saved to: {save_dir}"
    return outT, outR, msg


# ====== Gradio UI ======
with gr.Blocks(title="ALANet Reflection Removal - Single Image") as demo:
    gr.Markdown("# ALANet – Single Image Inference")
    gr.Markdown("GPU detected: **{}** | Checkpoints: `{}`".format(
        torch.cuda.device_count(), join(settings.checkpoints_dir, settings.name)))

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input Image")
            cap_t = gr.Textbox(
                label="Caption T – Transmission Layer (optional)",
                placeholder="e.g., 'a person behind the window...'"
            )
            cap_r = gr.Textbox(
                label="Caption R – Reflection Layer (optional)",
                placeholder="e.g., 'reflection of street lights...'"
            )
            save_toggle = gr.Checkbox(label="Save outputs to disk", value=True)
            save_dir_name = gr.Textbox(label="Save subdir name (optional)", placeholder="single_infer")
            run_btn = gr.Button("Run Inference", variant="primary")
        with gr.Column():
            out_t = gr.Image(type="pil", label="Output Transmission (T)")
            out_r = gr.Image(type="pil", label="Output Reflection (R)")
            status = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=ui_single_infer,
        inputs=[inp, cap_t, cap_r, save_toggle, save_dir_name],
        outputs=[out_t, out_r, status]
    )
if __name__ == "__main__":
    # Launch Gradio (default port 7860 for internal network, can be changed)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)



