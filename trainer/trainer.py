import torch.nn as nn
import torch

from torch.autograd import Variable

from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_console

import os, random


class AvgScore:
    def __init__(self):
        self.value = 0
        self.count = 0
        
        self.history = []
    
    def add(self, x):
        self.history.append(x)
        
        self.value += x
        self.count += 1
    
    def avg(self):
        if self.count == 0:
            return 0
        return self.value / self.count


class Buffer:
    def __init__(self, size=100, device=torch.device("cuda:0")):
        self.size = size 
        self.buffer = []

        self.device = device

    def push_and_pop(self, data):
        data = data.cpu()
        r = []
        
        for el in data.data:
            el = torch.unsqueeze(el, 0)
            
            if len(self.buffer) < self.size:
                self.buffer.append(el)
                r.append(el)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.size - 1)
                    r.append(self.buffer[i])
                    self.buffer[i] = el
                else:
                    r.append(el)
        
        return torch.cat(r).to(self.device)


class CycleGAN_Trainer:
    def __init__(self, A2B, B2A, A, B, ipython=False):
        self.A2B = A2B; self.B2A = B2A;
        self.A = A; self.B = B;
        
        self.ipython = ipython

    def train(self, epochs=100, buffer_size=150, figsize=(8, 5)):
        torch.cuda.empty_cache()

        tqdm = tqdm_notebook if self.ipython else tqdm_console
    
        gen_losses = AvgScore()
        dis_losses = AvgScore()
        
        buffer_A = Buffer(buffer_size)
        buffer_B = Buffer(buffer_size)
        
        gen_A2B.train()
        gen_B2A.train()
        
        dis_A.train()
        dis_B.train()
        
        for epoch in range(epochs):
            torch.cuda.empty_cache()
            
            tq = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch + 1}")
            gen_losses_epoch = AvgScore()
            dis_losses_epoch = AvgScore()

            for batch in tq:
                torch.cuda.empty_cache()
                
                A = Variable(input_A.copy_(batch[0]))
                B = Variable(input_B.copy_(batch[1]))
                
                batch_size = A.shape[0]

                optimizer_GA2B.zero_grad()
                optimizer_GB2A.zero_grad()
                
                identity_image_A = gen_B2A(A)
                identity_loss_A = criterion_identity(identity_image_A, A)
                
                identity_image_B = gen_A2B(B)
                identity_loss_B = criterion_identity(identity_image_B, B)
                
                identity_loss = (identity_loss_A + identity_loss_B) / 2
                
                fake_image_A = gen_B2A(B)
                fake_output_A = dis_A(fake_image_A).view(-1)
                loss_GAN_B2A = criterion_adversarial(fake_output_A, real_label)
                
                fake_image_B = gen_A2B(A)
                fake_output_B = dis_B(fake_image_B).view(-1)
                loss_GAN_A2B = criterion_adversarial(fake_output_B, real_label)
                
                loss_gan = (loss_GAN_B2A + loss_GAN_A2B) / 2

                recovered_image_A = gen_B2A(fake_image_B)
                loss_cycle_ABA = criterion_cycle(recovered_image_A, A)

                recovered_image_B = gen_A2B(fake_image_A)
                loss_cycle_BAB = criterion_cycle(recovered_image_B, B)
                
                loss_cycle = (loss_cycle_ABA + loss_cycle_BAB) / 2
                
                loss_G = (loss_gan + loss_cycle * k1 + identity_loss * k2) / 3
                loss_G.backward()
                
    #             nn.utils.clip_grad_norm_(gen_A2B.parameters(), 2.0)
    #             nn.utils.clip_grad_norm_(gen_B2A.parameters(), 2.0)

                set_requires_grad([dis_A, dis_B], False)
                
                optimizer_GA2B.step()
                optimizer_GB2A.step()

                set_requires_grad([dis_A, dis_B], True)
                
                optimizer_DA.zero_grad()
                
                real_output_A = dis_A(A).view(-1)
                loss_D_real_A = criterion_adversarial(real_output_A, real_label)
                
                fake_image_A = buffer_A.push_and_pop(fake_image_A)
                
                fake_output_A = dis_A(fake_image_A)
                loss_D_fake_A = criterion_adversarial(fake_output_A, fake_label)
                
                loss_DA = (loss_D_real_A + loss_D_fake_A) / 2
                loss_DA.backward()
                
    #             nn.utils.clip_grad_norm_(dis_A.parameters(), 2.0)
                optimizer_DA.step()
                
                optimizer_DB.zero_grad()
                
                real_output_B = dis_B(B).view(-1)
                loss_D_real_B = criterion_adversarial(real_output_B, real_label)
                
                fake_image_B = buffer_B.push_and_pop(fake_image_B)
                
                fake_output_B = dis_B(fake_image_B)
                loss_D_fake_B = criterion_adversarial(fake_output_B, fake_label)
                
                loss_DB = (loss_D_real_B + loss_D_fake_B) / 2
                loss_DB.backward()
                
    #             nn.utils.clip_grad_norm_(dis_B.parameters(), 2.0)
                optimizer_DB.step()
                
                gen_losses_epoch.add(loss_G.item())
                dis_losses_epoch.add((loss_DA.item() + loss_DB.item()) / 2)
                
                tq.set_postfix({
                    "gen_loss": gen_losses_epoch.avg(),
                    "dis_loss": dis_losses_epoch.avg(),
                    "dis_real_score_A": real_output_A.mean().item(),
                    "dis_fake_score_A": fake_output_A.mean().item(),
                    "dis_real_score_B": real_output_B.mean().item(),
                    "dis_fake_score_B": fake_output_B.mean().item(),
                })
            
            gen_losses.add(gen_losses_epoch.avg())
            dis_losses.add(dis_losses_epoch.avg())
            show_sample(figsize=figsize)

            lr_scheduler_GA2B.step()
            lr_scheduler_GB2A.step()
            print(f"lr_A2B: {get_lr(optimizer_GA2B)}; lr_B2A: {get_lr(optimizer_GB2A)};")