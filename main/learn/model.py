import torch
import torch.nn as nn
global mu
global var

class LinBnDrop(nn.Sequential):
    """Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"""
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

        
class Encoder(nn.Module):
    """Encoder for CITE-seq data"""
    def __init__(self, nfeatures_rna=10703, nfeatures_pro=192, hidden_rna=185,  hidden_pro=15, z_dim=128):
        super().__init__()
        self.nfeatures_rna = nfeatures_rna
        self.nfeatures_pro = nfeatures_pro
        self.encoder_rna = LinBnDrop(nfeatures_rna, hidden_rna, p=0.2, act=nn.ReLU())
        self.encoder_protein = LinBnDrop(nfeatures_pro, hidden_pro, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_rna + hidden_pro, z_dim,  p=0.2, act=nn.ReLU())
        self.weights_rna = nn.Parameter(torch.rand((1,nfeatures_rna)) * 0.001, requires_grad=True)
        self.weights_adt = nn.Parameter(torch.rand((1,nfeatures_pro)) * 0.001, requires_grad=True)
        self.fc_mu = LinBnDrop(z_dim,z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim,z_dim, p=0.2)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        global mu
        global var
        x_rna = self.encoder_rna(x[:, :self.nfeatures_rna]*self.weights_rna)
        x_pro = self.encoder_protein(x[:, self.nfeatures_rna:]*self.weights_adt)
        x = torch.cat([x_rna, x_pro], 1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        x = self.reparameterize(mu, var)
        return x
    

class Decoder(nn.Module):
    """Decoder for for 2 modalities data (citeseq data and shareseq data) """
    def __init__(self, nfeatures_rna=10703, nfeatures_pro=192,  hidden_rna=185,  hidden_pro=15, z_dim=128):
        super().__init__()
        self.nfeatures_rna = nfeatures_rna
        self.nfeatures_pro = nfeatures_pro
        self.decoder1 = nn.Sequential(LinBnDrop(z_dim, nfeatures_rna, act=nn.ReLU()))
        self.decoder2 = nn.Sequential(LinBnDrop(z_dim, nfeatures_pro,  act=nn.ReLU()))

    def forward(self, x):
        x_rna = self.decoder1(x)
        x_adt = self.decoder2(x)
        x = torch.cat((x_rna,x_adt),1)
        return x

    
class CiteAutoencoder(nn.Module):
    def __init__(self, nfeatures_rna=0, nfeatures_pro=0,  hidden_rna=185,  hidden_pro=15, z_dim=20,classify_dim=17):
        """ Autoencoder for 2 modalities data (citeseq data and shareseq data) """
        super().__init__()
        self.encoder = Encoder(nfeatures_rna, nfeatures_pro, hidden_rna,  hidden_pro, z_dim)
        self.classify = nn.Sequential(nn.Linear(z_dim, classify_dim))
        self.decoder = Decoder(nfeatures_rna, nfeatures_pro, hidden_rna,  hidden_pro, z_dim)
        
    def forward(self, x):
        global mu
        global var
        x = self.encoder(x)
        x_cty = self.classify(x)
        x = self.decoder(x)
        return x, x_cty,mu,var
    

class Encoder_3modal(nn.Module):
    """Encoder for TEA-seq data"""
    def __init__(self, nfeatures_rna=10703, nfeatures_pro=192, nfeatures_atac=192, hidden_rna=185, hidden_pro=30,  hidden_atac=185, z_dim=128):
        super().__init__()
        self.nfeatures_rna = nfeatures_rna
        self.nfeatures_pro = nfeatures_pro
        self.nfeatures_atac = nfeatures_atac

        self.encoder_rna = nn.Sequential(LinBnDrop(nfeatures_rna, hidden_rna, p=0.2, act=nn.ReLU()))
        self.encoder_protein = nn.Sequential(LinBnDrop(nfeatures_pro, hidden_pro, p=0.2, act=nn.ReLU()))
        self.encoder_atac = nn.Sequential(LinBnDrop(nfeatures_atac, hidden_atac, p=0.2, act=nn.ReLU()))
        self.encoder = LinBnDrop(hidden_rna + hidden_pro +  hidden_atac, z_dim,  p=0.2, act=nn.ReLU())
        self.weights_rna = nn.Parameter(torch.rand((1,nfeatures_rna)) * 0.001, requires_grad=True)
        self.weights_adt = nn.Parameter(torch.rand((1,nfeatures_pro)) * 0.001, requires_grad=True)
        self.weights_atac = nn.Parameter(torch.rand((1,nfeatures_atac)) * 0.001, requires_grad=True)
        self.fc_mu = nn.Sequential( LinBnDrop(z_dim,z_dim))
        self.fc_var = nn.Sequential( LinBnDrop(z_dim,z_dim))
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        global mu
        global var
        x_rna = self.encoder_rna(x[:, :self.nfeatures_rna]*self.weights_rna)
        x_pro = self.encoder_protein(x[:, self.nfeatures_rna:(self.nfeatures_rna+ self.nfeatures_pro)]*self.weights_adt)
        x_atac = self.encoder_atac(x[:, (self.nfeatures_rna+ self.nfeatures_pro):]*self.weights_atac)
        x = torch.cat([x_rna, x_pro, x_atac], 1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        x = self.reparameterize(mu, var)
        return x,mu,var


class Decoder_3modal(nn.Module):
    """Decoder for  TEA-seq data"""
    def __init__(self, nfeatures_rna=10703, nfeatures_pro=192, nfeatures_atac=10000, hidden_rna=185, hidden_pro=30, hidden_atac=185, z_dim=100):
        super().__init__()
        self.nfeatures_rna = nfeatures_rna
        self.nfeatures_pro = nfeatures_pro
        self.nfeatures_atac = nfeatures_atac
        self.decoder1 = nn.Sequential(LinBnDrop(z_dim, nfeatures_rna,  act=nn.ReLU()))
        self.decoder2 = nn.Sequential(LinBnDrop(z_dim, nfeatures_pro,  act=nn.ReLU()))
        self.decoder3 = nn.Sequential(LinBnDrop(z_dim, nfeatures_atac, act=nn.ReLU()))

    def forward(self, x):
        x_rna = self.decoder1(x)
        x_adt = self.decoder2(x)
        x_atac = self.decoder3(x)
        x = torch.cat((x_rna,x_adt,x_atac),1)
        return x
     

class CiteAutoencoder_3modal(nn.Module):
    def __init__(self, nfeatures_rna=10000, nfeatures_pro=30, nfeatures_atac=10000, hidden_rna=185, hidden_pro=30,  hidden_atac=185, z_dim=100,classify_dim=17):
        """ Autoencoder for  TEA-seq data """
        super().__init__()
        self.encoder = Encoder_3modal(nfeatures_rna, nfeatures_pro, nfeatures_atac, hidden_rna,  hidden_pro, hidden_atac, z_dim)
        self.classify = nn.Linear(z_dim, classify_dim)
        self.decoder = Decoder_3modal(nfeatures_rna, nfeatures_pro, nfeatures_atac, hidden_rna,  hidden_pro, hidden_atac, z_dim)

    def forward(self, x):
        global mu
        global var
        x,mu,var = self.encoder(x)
        x_cty = self.classify(x)
        x = self.decoder(x)
        return x, x_cty, mu, var
    
