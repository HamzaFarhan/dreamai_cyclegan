
from dreamai.utils import *
from dreamai.model import *
from dreamai.dai_imports import*
from dreamai_cyclegan.models import networks

class DaiGAN(Network):
    def __init__(self,
                 g_x = None,
                 g_y = None,
                 d_x = None,
                 d_y = None,
                 model_type = 'cycle_gan',
                 lambda_x = 10.0,
                 lambda_y = 10.0,
                 lambda_idt = 0.5,
                 lr_gx = 0.0002,
                 lr_dx = 0.0002,
                 lr_gy = 0.0002,
                 lr_dy = 0.0002,
                 beta = 0.5,
                 criterion = nn.L1Loss(),
                 criterion_gan = networks.GANLoss('lsgan'),
                 criterion_cycle = nn.L1Loss(),
                 criterion_idt = nn.L1Loss(),
                 optimizer_g_x = optim.Adam,
                 optimizer_g_y = optim.Adam,
                 optimizer_d_x = optim.Adam,
                 optimizer_d_y = optim.Adam,
                 device = None,
                 best_validation_loss = None,
                 best_psnr = None,
                 best_model_file = 'best_gan.pth',
                 ):

        super().__init__(device=device)

        self.g_x = g_x.to(device)
        self.d_x = d_x.to(device)
        self.g_y = None
        self.d_y = None
        self.is_cycle = False
        type_print = 'GAN'
        if g_y is not None:
            self.g_y = g_y.to(device)
            self.d_y = d_y.to(device)
            self.is_cycle = True
            type_print = 'CycleGAN'

        print(f'{type_print} using {model_type} generator.')


        self.set_model_params(criterion=criterion, criterion_gan=criterion_gan, criterion_cycle=criterion_cycle, criterion_idt=criterion_idt,
                              optimizer_g_x=optimizer_g_x, optimizer_d_x=optimizer_d_x, optimizer_g_y=optimizer_g_y, optimizer_d_y=optimizer_d_y,
                              lambda_x=lambda_x, lambda_y=lambda_y, lambda_idt=lambda_idt,
                              lr_gx=lr_gx, lr_dx=lr_dx, lr_gy=lr_gy, lr_dy=lr_dy, beta=beta,
                              best_validation_loss=best_validation_loss, model_type=model_type, best_model_file=best_model_file, best_psnr=best_psnr)

        self.loss_names = ['g', 'd_x', 'g_x', 'cycle_x', 'idt_x', 'd_y', 'g_y', 'cycle_y', 'idt_y']
    
    def set_model_params(self, criterion=nn.L1Loss(), criterion_gan=networks.GANLoss('lsgan'), criterion_cycle=nn.L1Loss(), criterion_idt=nn.L1Loss(),
                         lambda_x=10.0, lambda_y=10.0, lambda_idt=0.5, lr_gx=0.0002, lr_dx=0.0002, lr_gy=0.0002, lr_dy=0.0002, beta=0.5,
                         model_type='cycle_gan', best_psnr=None,
                         optimizer_g_x=optim.Adam, optimizer_d_x=optim.Adam, optimizer_g_y=optim.Adam, optimizer_d_y=optim.Adam,
                         best_model_file='best_cycle_gan.pth', best_validation_loss=None):
        
        self.best_psnr = best_psnr
        self.model_type = model_type
        self.best_model_file = best_model_file
        self.best_validation_loss = best_validation_loss
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_idt = lambda_idt
        # define loss functions
        self.criterion = criterion
        self.criterionGAN = criterion_gan.to(self.device)  # define GAN loss.
        self.criterionCycle = criterion_cycle
        self.criterionIdt = criterion_idt
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_g_x = optimizer_g_x(self.g_x.parameters(), lr=lr_gx, betas=(beta, 0.999))
        self.optimizer_d_x = optimizer_d_x(self.d_x.parameters(), lr=lr_dx, betas=(beta, 0.999))
        if self.is_cycle:
            self.optimizer_g_y = optimizer_g_y(self.g_y.parameters(), lr=lr_gy, betas=(beta, 0.999))
            self.optimizer_d_y = optimizer_d_y(self.d_y.parameters(), lr=lr_dy, betas=(beta, 0.999))


    def set_lrs(self, lr_gx=0.0002, lr_dx=0.0002, lr_gy=0.0002, lr_dy=0.0002):
        self.optimizer_g_x.param_groups[0]['lr'] = lr_gx
        self.optimizer_d_x.param_groups[0]['lr'] = lr_dx
        self.optimizer_g_y.param_groups[0]['lr'] = lr_gy
        self.optimizer_d_y.param_groups[0]['lr'] = lr_dy

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def domain_shift(self, inputs):
        self.eval()
        self.g_x.eval()
        self.g_x = self.g_x.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            # outputs = self.g_x[0](inputs)
            outputs = self.g_x(inputs)
        return outputs

    def forward(self, batch):
        
        self.real_x = batch[0].to(self.device)
        self.real_y = batch[1].to(self.device)
        self.fake_y = self.g_x(self.real_x)  # g_x(x)

        if self.is_cycle:
            self.rec_x = self.g_y(self.fake_y)   # g_y(g_x(x))
            self.fake_x = self.g_y(self.real_y)  # g_y(y)
            self.rec_y = self.g_x(self.fake_x)   # g_x(g_y(y))

    def backward_d_basic(self, net_d, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            net_d (network)      -- the discriminator d
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_d.backward() to calculate the gradients.
        """
        # Real
        pred_real = net_d(real)
        loss_d_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = net_d(fake.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        return loss_d

    def backward_d_x(self):
        self.loss_d_x = self.backward_d_basic(self.d_x, self.real_y, self.fake_y)

    def backward_d_y(self):
        self.loss_d_y = self.backward_d_basic(self.d_y, self.real_x, self.fake_x)

    def backward_g(self):
        """Calculate the loss for generators g_x and g_y"""
        lambda_idt = self.lambda_idt
        lambda_x = self.lambda_x
        lambda_y = self.lambda_y
        # Identity loss
        if lambda_idt > 0:
            # g_x should be identity if real_y is fed: ||g_x(y) - y||
            self.idt_x = self.g_x(self.real_y)
            self.loss_idt_x = self.criterionIdt(self.idt_x, self.real_y) * lambda_y * lambda_idt
            # g_y should be identity if real_x is fed: ||g_y(x) - x||
            self.idt_y = self.g_y(self.real_x)
            self.loss_idt_y = self.criterionIdt(self.idt_y, self.real_x) * lambda_x * lambda_idt
        else:
            self.loss_idt_x = 0
            self.loss_idt_y = 0

        # GAN loss d_x(g_x(x))
        self.loss_g_x = self.criterionGAN(self.d_x(self.fake_y), True)
        self.loss_g = self.loss_g_x
        if self.is_cycle:
            # GAN loss d_y(g_y(y))
            self.loss_g_y = self.criterionGAN(self.d_y(self.fake_x), True)
            # Forward cycle loss || g_y(g_x(x)) - x||
            self.loss_cycle_x = self.criterionCycle(self.rec_x, self.real_x) * lambda_x
            # self.loss_cycle_x = self.criterion(self.rec_x, self.real_x) * lambda_x
            # Backward cycle loss || g_x(g_y(y)) - y||
            self.loss_cycle_y = self.criterionCycle(self.rec_y, self.real_y) * lambda_y
            # self.loss_cycle_y = self.criterion(self.rec_y, self.real_y) * lambda_y
            # combined loss and calculate gradients
            self.loss_g = self.loss_g_x + self.loss_g_y + self.loss_cycle_x + self.loss_cycle_y + self.loss_idt_x + self.loss_idt_y
        self.loss_g.backward()

    def optimize_parameters(self, batch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(batch)      # compute fake images and reconstruction images.
        # g_x and g_y
        self.set_requires_grad([self.d_x], False)  # ds require no gradients when optimizing Gs
        self.optimizer_g_x.zero_grad()  # set g_x and g_y's gradients to zero
        if self.is_cycle:
            self.set_requires_grad([self.d_y], False)
            self.optimizer_g_y.zero_grad()  # set g_x and g_y's gradients to zero
        self.backward_g()             # calculate gradients for g_x and g_y
        self.optimizer_g_x.step()       # update g_x and g_y's weights
        if self.is_cycle:
            self.optimizer_g_y.step()
        # d_x and d_y
        self.set_requires_grad([self.d_x], True)
        self.optimizer_d_x.zero_grad()
        self.backward_d_x()
        self.optimizer_d_x.step()
        if self.is_cycle:
            self.set_requires_grad([self.d_y], True)
            self.optimizer_d_y.zero_grad()
            self.backward_d_y()
            self.optimizer_d_y.step()

    def batch_to_loss(self,data_batch):
        self.optimize_parameters(data_batch)
        return self.loss_g.item()

    def fit(self, trainloader, validloader, cycle_len=2, num_cycles=1, print_every=10,
            validate_every=1, save_best_every=1, load_best=False,
            eval_thresh=0.5, saving_crit='loss'):
        # os.makedirs('saved_weights', exist_ok=True)
        weights_folder = Path('saved_weights')
        epochs = cycle_len
        optim_path = Path(self.best_model_file)
        optim_path = optim_path.stem + '_optim' + optim_path.suffix
        # lr = self.optimizer_g.param_groups[0]['lr']
        with mlflow.start_run() as run:
            for cycle in range(num_cycles):
                for epoch in range(epochs):
                    print(f'Cycle: {cycle+1}/{num_cycles}')
                    print('Epoch:{:3d}/{}\n'.format(epoch+1,epochs))
                    mlflow.log_param('epochs',epochs)
                    mlflow.log_param('lr',self.optimizer_g_x.param_groups[0]['lr'])
                    mlflow.log_param('bs',trainloader.batch_size)
                    epoch_train_loss =  self.train_((epoch,epochs), trainloader, print_every)
                            
                    if  validate_every and (epoch % validate_every == 0):
                        if self.model_type == 'enhancement':
                            self.visual_eval(validloader)
                        else:
                            t2 = time.time()
                            eval_dict = self.evaluate(validloader,thresh=eval_thresh)
                            epoch_validation_loss = eval_dict['final_loss']
                            mlflow.log_metric('Train Loss',epoch_train_loss)
                            mlflow.log_metric('Validation Loss',epoch_validation_loss)
                            time_elapsed = time.time() - t2
                            if time_elapsed > 60:
                                time_elapsed /= 60.
                                measure = 'min'
                            else:
                                measure = 'sec'    
                            print('\n'+'/'*36+'\n'
                                    f"{time.asctime().split()[-2]}\n"
                                    f"Epoch {epoch+1}/{epochs}\n"    
                                    f"Validation time: {time_elapsed:.6f} {measure}\n"    
                                    f"Epoch training loss: {epoch_train_loss:.6f}\n"                        
                                    f"Epoch validation loss: {epoch_validation_loss:.6f}"
                                )

                        if self.model_type == 'super_res':# or self.model_type == 'enhancement':
                            epoch_psnr = eval_dict['psnr']
                            mlflow.log_metric('Validation PSNR',epoch_psnr)
                            print("Validation psnr: {:.3f}".format(epoch_psnr))

                        print('\\'*36+'\n')

                        if saving_crit == 'loss':
                            if self.best_validation_loss == None or (epoch_validation_loss <= self.best_validation_loss):
                                print('\n**********Updating best validation loss**********\n')
                                if self.best_validation_loss is not None:
                                    print('Previous best: {:.7f}'.format(self.best_validation_loss))
                                print('New best loss = {:.7f}\n'.format(epoch_validation_loss))
                                print('*'*49+'\n')
                                self.best_validation_loss = epoch_validation_loss
                                mlflow.log_metric('Best Loss',self.best_validation_loss)

                                best_gx_path, best_gy_path, optim_g_x_path, optim_g_y_path = self.save_model(epoch_validation_loss, epoch+1,
                                weights_folder, mlflow_saved_folder='mlflow_saved_training_models',
                                mlflow_logged_folder='mlflow_logged_models')

                        elif saving_crit == 'psnr':
                            if self.best_psnr == None or (epoch_psnr >= self.best_psnr):
                                print('\n**********Updating best psnr**********\n')
                                if self.psnr is not None:
                                    print('Previous best: {:.7f}'.format(self.best_psnr))
                                print('New best psnr = {:.7f}\n'.format(epoch_psnr))
                                print('*'*49+'\n')
                                self.best_psnr = epoch_psnr
                                mlflow.log_metric('Best Psnr',self.best_psnr)

                                best_gx_path, best_gy_path, optim_g_x_path, optim_g_y_path = self.save_model(epoch_psnr, epoch+1,
                                weights_folder, mlflow_saved_folder='mlflow_saved_training_models',
                                mlflow_logged_folder='mlflow_logged_models')

                        self.train()
        torch.cuda.empty_cache()
        if load_best:
            try:
                print('\nLoaded best model\n')
                self.g_x.load_state_dict(torch.load(best_gx_path))
                self.optimizer_g_x.load_state_dict(torch.load(optim_g_x_path))
                if self.is_cycle:
                    self.g_y.load_state_dict(torch.load(best_gy_path))
                    self.optimizer_g_y.load_state_dict(torch.load(optim_g_y_path))
                # os.remove(self.best_model_file)
                # os.remove(optim_path)
            except:
                pass

    def train_(self, e, trainloader, print_every):

        self.train()
        epoch,epochs = e
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            batches += 1
            loss = self.batch_to_loss(data_batch)
            running_loss += loss
            if batches % print_every == 0:
                elapsed = time.time()-t1
                if elapsed > 60:
                    elapsed /= 60.
                    measure = 'min'
                else:
                    measure = 'sec'
                batch_time = time.time()-t0
                if batch_time > 60:
                    batch_time /= 60.
                    measure2 = 'min'
                else:
                    measure2 = 'sec'    
                print('+----------------------------------------------------------------------+\n'
                        f"{time.asctime().split()[-2]}\n"
                        f"Time elapsed: {elapsed:.3f} {measure}\n"
                        f"Epoch:{epoch+1}/{epochs}\n"
                        f"Batch: {batches+1}/{len(trainloader)}\n"
                        f"Batch training time: {batch_time:.3f} {measure2}\n"
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader) 

    def compute_loss(self,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        loss = self.criterion(outputs, labels)
        ret['overall_loss'] = loss
        return loss,ret
    
    def save_model(self, crit='', epoch='', weights_folder='saved_weights',
                   mlflow_saved_folder='mlflow_saved_training_models', mlflow_logged_folder='mlflow_logged_models'):
            weights_folder = Path(weights_folder)
            os.makedirs(weights_folder, exist_ok=True)
            if type(epoch) != str:
                epoch = str(epoch)
            if type(crit) != str:
                crit = str(round(crit,3))
            curr_time = str(datetime.now())
            curr_time = '_'+curr_time.split()[1].split('.')[0]
            suff = Path(self.best_model_file).suffix
            best_model_file = Path(self.best_model_file).stem+f'_{crit}_{epoch+curr_time}'

            best_gx_file = Path(self.best_model_file).stem+f'_gx_{crit}_{epoch+curr_time}'
            best_gy_file = Path(self.best_model_file).stem+f'_gy_{crit}_{epoch+curr_time}'
            best_gx_path = weights_folder/(best_gx_file + suff)
            best_gy_path = weights_folder/(best_gy_file + suff)

            optim_g_x_path = weights_folder/(best_model_file + '_g_x_optim' + suff)
            optim_g_y_path = weights_folder/(best_model_file + '_g_y_optim' + suff)

            torch.save(self.g_x.state_dict(), best_gx_path)
            torch.save(self.optimizer_g_x.state_dict(),optim_g_x_path)
            if self.is_cycle:
                torch.save(self.g_y.state_dict(), best_gy_path)
                torch.save(self.optimizer_g_y.state_dict(),optim_g_y_path)

            mlflow.pytorch.log_model(self,mlflow_logged_folder)
            mlflow_save_path = Path(mlflow_saved_folder)/best_model_file
            mlflow.pytorch.save_model(self,mlflow_save_path)
            return best_gx_path, best_gy_path, optim_g_x_path, optim_g_y_path

    def visual_eval(self, dataloader, save=True):

        data_batch = random.choice(dataloader.dataset)
        img, target_domain = data_batch[0],data_batch[1]
        img = img.unsqueeze(0).to(self.device)
        gen_domain = self.domain_shift(img)
        torchvision.utils.save_image([
                                    #   denorm_tensor(target_domain.cpu()[0], self.img_mean, self.img_std),
                                    img.cpu()[0],
                                    target_domain,
                                    gen_domain.cpu()[0]
                                    ],
                                    fp='current_cyclegan_performance.png')
        if save:
            self.save_model()

    def evaluate(self,dataloader, **kwargs):

        # res = self.residual
        # self.set_residual(False)
        running_loss = 0.
        running_psnr = 0.
        rmse_ = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in dataloader:
                try:
                    img, target_domain, hr_resized = data_batch[0],data_batch[1],data_batch[2]
                except:
                    img, target_domain = data_batch[0],data_batch[1]
                img = img.to(self.device)
                target_domain = target_domain.to(self.device)
                gen_domain = self.domain_shift(img)
                _,loss_dict = self.compute_loss(gen_domain,target_domain)
                try:
                    torchvision.utils.save_image([
                                                #   denorm_tensor(target_domain.cpu()[0], self.img_mean, self.img_std),
                                                target_domain.cpu()[0],
                                                hr_resized[0],
                                                gen_domain.cpu()[0]
                                                ],
                                                fp='current_cyclegan_performance.png')
                except:
                    torchvision.utils.save_image([
                                                #   denorm_tensor(target_domain.cpu()[0], self.img_mean, self.img_std),
                                                target_domain.cpu()[0],
                                                gen_domain.cpu()[0]
                                                ],
                                                fp='current_cyclegan_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(gen_domain,target_domain).cpu().numpy()
        # self.set_residual(res)
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

    # def predict(self,inputs,actv = None):
    #     res = self.residual
    #     self.set_residual(False)
    #     self.eval()
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         inputs = inputs.to(self.device)
    #         outputs = self.forward(inputs)
    #     if actv is not None:
    #         return actv(outputs)
    #     self.set_residual(res)
    #     return outputs
