import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

RANDOM_SEED = 32

# THIS FILE CONTAINS THE DEPRECATED CODE FOR THE MODELS
warnings.warn("This module is deprecated in favour of machine_learning", DeprecationWarning, stacklevel=2)


class MLP1(nn.Module):
    def __init__(self, init_dim, out_dim):
        super(MLP1, self).__init__()
        self.linear1 = nn.Linear(init_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, out_dim)
        self.drop_out1 = nn.Dropout(p=0.05)
        self.drop_out2 = nn.Dropout(p=0.05)
        self.drop_out3 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.drop_out1(F.relu(self.linear1(x)))
        x = self.drop_out2(F.relu(self.linear2(x)))
        x = self.drop_out3(F.relu(self.linear3(x)))
        x = self.linear4(x)
        return x


class MLP3(nn.Module):
    def __init__(self, init_dim, out_dim):
        super(MLP3, self).__init__()
        self.linear1 = nn.Linear(init_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 32)
        self.linear8 = nn.Linear(32, out_dim)
        self.drop_out1 = nn.Dropout(p=0.05)
        self.drop_out2 = nn.Dropout(p=0.05)
        self.drop_out3 = nn.Dropout(p=0.05)
        self.drop_out4 = nn.Dropout(p=0.05)
        self.drop_out5 = nn.Dropout(p=0.05)
        self.drop_out6 = nn.Dropout(p=0.05)
        self.drop_out7 = nn.Dropout(p=0.05)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.drop_out1(self.act(self.linear1(x)))
        x = self.drop_out2(self.act(self.linear2(x)))
        x = self.drop_out3(self.act(self.linear3(x)))
        x = self.drop_out4(self.act(self.linear4(x)))
        x = self.drop_out5(self.act(self.linear5(x)))
        x = self.drop_out6(self.act(self.linear6(x)))
        x = self.drop_out7(self.act(self.linear7(x)))
        x = self.linear8(x)
        return x


##############################################################

#               COMBINED MODEL STARTS FROM HERE

##############################################################
class DeepBackbone(nn.Module):
    def __init__(self, init_dim):
        super(DeepBackbone, self).__init__()

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        self.out_dim = 128

        self.linear1 = nn.Linear(init_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, self.out_dim)

        self.drop_out1 = nn.Dropout(p=0.05)
        self.drop_out2 = nn.Dropout(p=0.05)
        self.drop_out3 = nn.Dropout(p=0.05)
        self.drop_out4 = nn.Dropout(p=0.05)
        self.drop_out5 = nn.Dropout(p=0.05)
        self.drop_out6 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.drop_out1(F.relu(self.linear1(x)))
        x = self.drop_out2(F.relu(self.linear2(x)))
        x = self.drop_out3(F.relu(self.linear3(x)))
        x = self.drop_out4(F.relu(self.linear4(x)))
        x = self.drop_out5(F.relu(self.linear5(x)))
        x = self.drop_out6(F.relu(self.linear6(x)))

        return x


class InBetweenBackbone(nn.Module):
    def __init__(self, init_dim):
        super(InBetweenBackbone, self).__init__()

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        self.out_dim = 64

        self.linear1 = nn.Linear(init_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, self.out_dim)
        self.drop_out1 = nn.Dropout(p=0.05)
        self.drop_out2 = nn.Dropout(p=0.05)
        self.drop_out3 = nn.Dropout(p=0.05)
        self.drop_out4 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.drop_out1(F.relu(self.linear1(x)))
        x = self.drop_out2(F.relu(self.linear2(x)))
        x = self.drop_out3(F.relu(self.linear3(x)))
        x = self.drop_out4(F.relu(self.linear4(x)))
        return x


class ShallowBackbone(nn.Module):
    def __init__(self, init_dim):
        super(ShallowBackbone, self).__init__()

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        self.out_dim = 32

        self.linear1 = nn.Linear(init_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, self.out_dim)

        self.drop_out1 = nn.Dropout(p=0.05)
        self.drop_out2 = nn.Dropout(p=0.05)
        self.drop_out3 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.drop_out1(F.relu(self.linear1(x)))
        x = self.drop_out2(F.relu(self.linear2(x)))
        x = self.drop_out3(F.relu(self.linear3(x)))

        return x


class CombinedModel(nn.Module):
    def __init__(self, tabular_size, out_dim, freeze_backbone, device):

        super(CombinedModel, self).__init__()

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)

        self.freeze_backbone = freeze_backbone
        self.backbone_dict = {
            "Deep": DeepBackbone(tabular_size).to(device),
            "Shallow": ShallowBackbone(tabular_size).to(device),
            "MiddleLog": InBetweenBackbone(tabular_size).to(device),
            "MiddleExtracted": InBetweenBackbone(tabular_size).to(device),
        }
        for freezing in self.freeze_backbone:
            if freezing in self.backbone_dict.keys():
                del self.backbone_dict[freezing]
        self.total_out = sum([out_value.out_dim for out_value in self.backbone_dict.values()])

        self.linear6 = nn.Linear(self.total_out, int(self.total_out / 2.0))
        self.linear7 = nn.Linear(int(self.total_out / 2.0), 32)
        self.linear8 = nn.Linear(32, out_dim)

        self.drop_out6 = nn.Dropout(p=0.05)
        self.drop_out7 = nn.Dropout(p=0.05)

    def forward(self, tabular_x_deep, tabular_x_shallow, log_in_x, extracted_in_x):
        x_final = []
        if "Deep" in self.backbone_dict.keys():
            deep_x = self.backbone_dict["Deep"](tabular_x_deep)
            x_final.append(deep_x)
        if "Shallow" in self.backbone_dict.keys():
            shallow_x = self.backbone_dict["Shallow"](tabular_x_shallow)
            x_final.append(shallow_x)
        if "MiddleLog" in self.backbone_dict.keys():
            log_x = self.backbone_dict["MiddleLog"](log_in_x)
            x_final.append(log_x)
        if "MiddleExtracted" in self.backbone_dict.keys():
            extracted_x = self.backbone_dict["MiddleExtracted"](extracted_in_x)
            x_final.append(extracted_x)
        x = torch.cat(x_final, dim=1)
        x = self.drop_out6(F.relu(self.linear6(x)))
        x = self.drop_out7(F.relu(self.linear7(x)))
        x = self.linear8(x)
        return x
