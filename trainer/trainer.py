import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
import os
import sys
from PIL import Image
import torchvision
from tqdm import tqdm
from data_loader.loader import ContentData
import torch.distributed as dist
import torch.nn.functional as F

class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader, 
                logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        self.model = unet
        self.diffusion = diffusion
        self.vae = vae
        self.recon_criterion = criterion['recon']
        self.nce_criterion = criterion['nce']
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.ocr_model = ocr_model
        self.ctc_criterion = ctc_loss
        self.device = device
      
    def _train_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device)
        
        # vae encode
        images = self.vae.encode(images).latent_dist.sample()
        images = images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        
       
        predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _finetune_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid, target, target_lengths = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device), \
            data['target'].to(self.device), \
            data['target_lengths'].to(self.device)
        
        # vae encode
        latent_images = self.vae.encode(images).latent_dist.sample()
        latent_images = latent_images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True).to(self.device)
        x_t, noise = self.diffusion.noise_images(latent_images, t)
        
        x_start, predicted_noise, high_nce_emb, low_nce_emb = self.diffusion.train_ddim(self.model, x_t, style_ref, laplace_ref,
                                                        content_ref, t, sampling_timesteps=5)
 
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        rec_out = self.ocr_model(x_start)
        input_lengths = torch.IntTensor(x_start.shape[0]*[rec_out.shape[0]])
        ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target, input_lengths, target_lengths)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss + 0.1*ctc_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item(), "ctc_loss": ctc_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save(path)
        return im

    @torch.no_grad()
    def _valid_iter(self, epoch):
        print('loading test dataset, the number is', len(self.valid_data_loader))
        self.model.eval()
        # use the first batch of dataloader in all validations for better visualization comparisons
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)
        # prepare input
        images, style_ref, laplace_ref, content_ref = test_data['img'].to(self.device), \
            test_data['style'].to(self.device), \
            test_data['laplace'].to(self.device), \
            test_data['content'].to(self.device)
    
        load_content = ContentData()
        # forward
        texts = ['getting', 'both', 'success']
        for text in texts:
            rank = dist.get_rank()
            text_ref = load_content.get_content(text)
            text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (text_ref.shape[1]*32)//8)).to(self.device)
            preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref, laplace_ref, text_ref)
            out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{rank}.png")
            self._save_images(preds, out_path)

    def train(self):
        """start training iterations"""
        for epoch in range(cfg.SOLVER.EPOCHS):
            self.data_loader.sampler.set_epoch(epoch)
            print(f"Epoch:{epoch} of process {dist.get_rank()}")
            dist.barrier()
            if dist.get_rank() == 0:
                pbar = tqdm(self.data_loader, leave=False)
            else:
                pbar = self.data_loader

            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        if dist.get_rank() == 0:
                            self._save_checkpoint(total_step)
                    else:
                        pass
                    if self.valid_data_loader is not None:
                        if (total_step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (total_step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                            self._valid_iter(total_step)
                        else:
                            pass 
                else:
                    self._train_iter(data, total_step, pbar)

            if (epoch+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (epoch+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(epoch)
                else:
                    pass
            if self.valid_data_loader is not None:
                if (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN  and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(epoch)
            else:
                pass

            if dist.get_rank() == 0:
                pbar.close()

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_model_dir, str(epoch)+'-'+"ckpt.pt"))