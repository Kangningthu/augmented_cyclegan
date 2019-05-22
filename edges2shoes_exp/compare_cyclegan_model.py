class StochCycleGAN(object):
    """Stochastic cycle gan"""
    def __init__(self, opt, ignore_noise=False, testing=False):

        self.ignore_noise = ignore_noise

        ##### model options
        self.old_lr = opt.lr
        opt.use_sigmoid = opt.no_lsgan

        self.opt = opt

        ##### define all networks we need here
        self.netG_A_B = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     which_model_netG=opt.which_model_netG,
                                                     norm=opt.norm, use_dropout=opt.use_dropout,
                                                     gpu_ids=opt.gpu_ids)

        self.netG_B_A = networks.define_G(input_nc=opt.output_nc,
                                          output_nc=opt.input_nc, ngf=opt.ngf,
                                          which_model_netG=opt.which_model_netG,
                                          norm=opt.norm, use_dropout=opt.use_dropout,
                                          gpu_ids=opt.gpu_ids)

        self.netD_A = networks.define_D_A(input_nc=opt.input_nc,
                                          ndf=32, which_model_netD=opt.which_model_netD,
                                          norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

        self.netD_B = networks.define_D_B(input_nc=opt.output_nc,
                                          ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                          norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

        ##### define all optimizers here
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_B.parameters(),
                                                            self.netG_B_A.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                                                            self.netD_B.parameters()),
                                            lr=opt.lr/5., betas=(opt.beta1, 0.999))

        self.criterionGAN = functools.partial(criterion_GAN, use_sigmoid=opt.use_sigmoid)
        self.criterionCycle = F.l1_loss

        if not testing:
            with open("%s/nets.txt" % opt.expr_dir, 'w') as nets_f:
                networks.print_network(self.netG_A_B, nets_f)
                networks.print_network(self.netG_B_A, nets_f)
                networks.print_network(self.netD_A, nets_f)
                networks.print_network(self.netD_B, nets_f)

    def train_instance(self, real_A, real_B, prior_z_B):

        if self.ignore_noise:
            prior_z_B = prior_z_B.mul(0.).add(1.)

        ##### genearte B using A and z_B : A ==> B <== z_B
        fake_B = self.netG_A_B.forward(real_A, prior_z_B)

        ##### infer A from B : A <-- B
        fake_A = self.netG_B_A.forward(real_B)

        ##### Discriminator A loss and optimization
        # NOTE: ".detach()" makes sure no gradient flows to the generator or encoder
        pred_fake_A = self.netD_A.forward(fake_A.detach())
        loss_D_fake_A = self.criterionGAN(pred_fake_A, False)

        pred_true_A = self.netD_A.forward(real_A)
        loss_D_true_A = self.criterionGAN(pred_true_A, True)

        pred_fake_B = self.netD_B.forward(fake_B.detach())
        loss_D_fake_B = self.criterionGAN(pred_fake_B, False)

        pred_true_B = self.netD_B.forward(real_B)
        loss_D_true_B = self.criterionGAN(pred_true_B, True)

        loss_D_A = 0.5 * (loss_D_fake_A + loss_D_true_A)
        loss_D_B = 0.5 * (loss_D_fake_B + loss_D_true_B)
        loss_D = loss_D_A + loss_D_B

        # NOTE: after the following snippet, the discriminator parameters will change
        self.optimizer_D.zero_grad()

        loss_D.backward()
        gnorm_D_A = torch.nn.utils.clip_grad_norm(self.netD_A.parameters(), self.opt.max_gnorm)
        gnorm_D_B = torch.nn.utils.clip_grad_norm(self.netD_B.parameters(), self.opt.max_gnorm)

        self.optimizer_D.step()

        ##### Generator and Encoder ALI loss
        # NOTE: The generator and encoder ALI loss is computed using the new (updated)
        # discriminator parameters.
        pred_fake_A = self.netD_A.forward(fake_A)
        loss_G_A = self.criterionGAN(pred_fake_A, True)

        pred_fake_B = self.netD_B.forward(fake_B)
        loss_G_B = self.criterionGAN(pred_fake_B, True)

        ##### A -> B -> A cycle loss
        rec_A = self.netG_B_A.forward(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, real_A)

        ##### B -> A -> B cycle loss
        rec_B = self.netG_A_B.forward(fake_A, prior_z_B)
        loss_cycle_B = self.criterionCycle(rec_B, real_B)

        ##### Generation and Encoder optimization
        loss_cycle = loss_cycle_A * self.opt.lambda_A + loss_cycle_B * self.opt.lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle

        self.optimizer_G.zero_grad()
        loss_G.backward()
        gnorm_G_A_B = torch.nn.utils.clip_grad_norm(self.netG_A_B.parameters(), self.opt.max_gnorm)
        gnorm_G_B_A = torch.nn.utils.clip_grad_norm(self.netG_B_A.parameters(), self.opt.max_gnorm)

        self.optimizer_G.step()

        ##### Return dicts
        losses  = OrderedDict([('D_A', loss_D_A.data[0]), ('G_A', loss_G_A.data[0]), ('Cyc_A', loss_cycle_A.data[0]),
                               ('D_B', loss_D_B.data[0]), ('G_B', loss_G_B.data[0]), ('Cyc_B', loss_cycle_B.data[0]),
                               ('P_t_A', pred_true_A.data.mean()), ('P_f_A', pred_fake_A.data.mean()),
                               ('P_t_B', pred_true_B.data.mean()), ('P_f_B', pred_fake_B.data.mean())])

        visuals = OrderedDict([('real_A', real_A.data), ('fake_B', fake_B.data), ('rec_A', rec_A.data),
                               ('real_B', real_B.data), ('fake_A', fake_A.data), ('rec_B', rec_B.data)])

        if self.opt.monitor_gnorm:
            gnorms = OrderedDict([('gnorm_G_A_B', gnorm_G_A_B),
                                  ('gnorm_G_B_A', gnorm_G_B_A),
                                  ('gnorm_D_B', gnorm_D_B),
                                  ('gnorm_D_A', gnorm_D_A),])
            return losses, visuals, gnorms

        return losses, visuals

    def generate_cycle(self, real_A, real_B, prior_z_B):
        if self.ignore_noise:
            prior_z_B = prior_z_B.mul(0.).add(1.)

        fake_B = self.netG_A_B.forward(real_A, prior_z_B)
        fake_A = self.netG_B_A.forward(real_B)
        rec_A = self.netG_B_A.forward(fake_B)
        rec_B = self.netG_A_B.forward(fake_A, prior_z_B)
        visuals = OrderedDict([('real_A', real_A.data), ('fake_B', fake_B.data), ('rec_A', rec_A.data),
                               ('real_B', real_B.data), ('fake_A', fake_A.data), ('rec_B', rec_B.data)])
        return visuals

    def generate_multi_cycle(self, real_B, steps):
        images = [real_B.data]
        B = real_B
        for i in range(steps):
            A = self.netG_B_A.forward(B)
            z_B = Variable(real_B.data.new(real_B.size(0), self.opt.nlatent, 1, 1).normal_(0, 1),
                           volatile=True)
            if self.ignore_noise:
                z_B = z_B.mul(0.).add(1.)
            B = self.netG_A_B.forward(A, z_B)
            images.extend([A.data, B.data])
        return images

    def generate_cycle_B_multi(self, real_B, multi_prior_z_B):
        fake_A = self.netG_B_A.forward(real_B)
        size = real_B.size()
        num = multi_prior_z_B.size(0) // real_B.size(0)

        multi_fake_A = fake_A.unsqueeze(1).repeat(1, num, 1, 1, 1)
        multi_fake_A = multi_fake_A.view(size[0]*num,size[1],size[2],size[3])

        multi_fake_B = self.netG_A_B.forward(multi_fake_A, multi_prior_z_B)

        return fake_A, multi_fake_B

    def generate_noisy_cycle(self, real_B, std):
        fake_A = self.netG_B_A.forward(real_B)
        z_B = Variable(real_B.data.new(real_B.size(0), self.opt.nlatent, 1, 1).normal_(0, 1),
                       volatile=True)
        if self.ignore_noise:
            z_B = z_B.mul(0.).add(1.)
        noise_std = std / 127.5
        noisy_fake_A = fake_A + Variable(fake_A.data.new(*fake_A.size()).normal_(0, noise_std),
                                         volatile=True)
        noisy_fake_A = torch.clamp(noisy_fake_A, -1, 1)
        rec_B = self.netG_A_B.forward(noisy_fake_A, z_B)
        return rec_B

    def predict_A(self, real_B):
        return self.netG_B_A.forward(real_B)

    def predict_B(self, real_A, z_B):
        if self.ignore_noise:
            z_B = z_B.mul(0.).add(1.)
        return self.netG_A_B.forward(real_A, z_B)

    def generate_multi(self, real_A, multi_prior_z_B):
        if self.ignore_noise:
            multi_prior_z_B = multi_prior_z_B.mul(0.).add(1.)

        size = real_A.size()
        num = multi_prior_z_B.size(0) // real_A.size(0)

        multi_real_A = real_A.unsqueeze(1).repeat(1, num, 1, 1, 1)
        multi_real_A = multi_real_A.view(size[0]*num,size[1],size[2],size[3])

        multi_fake_B = self.netG_A_B.forward(multi_real_A, multi_prior_z_B)

        return multi_fake_B

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, chk_name):
        chk_path = os.path.join(self.opt.expr_dir, chk_name)
        checkpoint = {
            'netG_A_B':self.netG_A_B.state_dict(),
            'netG_B_A':self.netG_B_A.state_dict(),
            'netD_A':self.netD_A.state_dict(),
            'netD_B':self.netD_B.state_dict(),
            'optimizer_D':self.optimizer_D.state_dict(),
            'optimizer_G':self.optimizer_G.state_dict()
        }
        torch.save(checkpoint, chk_path)

    def load(self, chk_path):
        checkpoint = torch.load(chk_path)

        self.netG_A_B.load_state_dict(checkpoint['netG_A_B'])
        self.netG_B_A.load_state_dict(checkpoint['netG_B_A'])
        self.netD_A.load_state_dict(checkpoint['netD_A'])
        self.netD_B.load_state_dict(checkpoint['netD_B'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    def eval(self):
        self.netG_A_B.eval()
        self.netG_B_A.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def train(self):
        self.netG_A_B.train()
        self.netG_B_A.train()
        self.netD_A.train()
        self.netD_B.train()