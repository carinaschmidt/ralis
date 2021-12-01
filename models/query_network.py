import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.parser as parser

class QueryNetworkDQN(nn.Module):
    def __init__(self, indexes_full_state=10 * 128, input_size=38, input_size_subset=38, sim_size=64): #192, 192, 160
        super(QueryNetworkDQN, self).__init__()
        self.conv1_s = nn.Conv1d(input_size_subset, 256, 1) #for ACDC: input_size_subset=198; agnostic 192; for BraTS18:
        self.bn1_s = nn.BatchNorm1d(input_size_subset)
        self.conv2_s = nn.Conv1d(256, 128, 1)
        self.bn2_s = nn.BatchNorm1d(256)
        self.conv3_s = nn.Conv1d(128, 1, 1)
        self.bn3_s = nn.BatchNorm1d(128)
        print("indexes full state in DQN: ", indexes_full_state)
        self.linear_s = nn.Linear(indexes_full_state, 128) #288 for ACDC; indexes_full_state = 160
        self.bn_last_s = nn.BatchNorm1d(int(indexes_full_state)) #indexes_full_state = 160

        self.conv1 = nn.Conv1d(input_size, 512, 1) #for ACDC: input_size_subset=198, agnostic: 192
        self.bn1 = nn.BatchNorm1d(input_size) #192
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv_final2 = nn.Conv1d(128 + 128, 1, 1)
        self.bn_final = nn.BatchNorm1d(128 + 128)

        self.conv_bias = nn.Conv1d(sim_size, 1, 1) #64
        self.bn_bias = nn.BatchNorm1d(sim_size)

        self.final_q = nn.Linear(256, 1)

        self.sim_size = sim_size

    def forward(self, x, subset):
        # x action, subset state
        # Compute state representation
        args = parser.get_arguments()
        print("len(subset): ", len(subset))
        if "agnostic" in args.exp_name:
            sub = subset.transpose(2, 1).contiguous()
            print("size of sub in forward: ", sub.size())
            sub = self.conv1_s(F.relu(self.bn1_s(sub))) #running_mean should contain 192 elements not 197
            sub = self.conv2_s(F.relu(self.bn2_s(sub)))
            sub = self.conv3_s(F.relu(self.bn3_s(sub)))
            sub = self.linear_s(F.relu(self.bn_last_s(sub.view(sub.size()[0], -1))))
            sub = sub.unsqueeze(2).repeat(1, 1, x.shape[1])

            #bias = self.conv_bias(F.relu(self.bn_bias(x[:, :, -self.sim_size:].transpose(1, 2).contiguous()))).transpose(1,
            #                                                                                                            2)

            # Compute action representation
            x = x.transpose(1, 2).contiguous()
            #print("self.sim_size: ", self.sim_size)
            #print("x: ", x)
            x = self.conv1(F.relu(self.bn1(x)))
            x = self.conv2(F.relu(self.bn2(x)))
            x = self.conv3(F.relu(self.bn3(x)))

            # Compute Q(s,a)
            out = torch.cat([x, sub], dim=1)
            out = self.conv_final2(self.bn_final(out))
            return (out.transpose(1, 2)).view(out.size()[0], -1)  #removed bias
        else:
            sub = subset.transpose(2, 1).contiguous()
            print("sub.size(): ", sub.size()) # torch.Size([16, 197, 288]), torch.Size([16, 197, 2560]), brats: torch.Size([16, 197, 7680]) ,brats [16,197, 960]
            sub = self.conv1_s(F.relu(self.bn1_s(sub))) #running_mean should contain 192 elements not 197 # brats: running_mean should contain 197 elements not 167
            sub = self.conv2_s(F.relu(self.bn2_s(sub)))
            sub = self.conv3_s(F.relu(self.bn3_s(sub)))
            print("sub.size before linear_s: ", sub.size()) #  torch.Size([16, 1, 288] , brats ([16, 1, 2560])
            #print("linear_s: ", self.linear_s) #[16,1,288]
            #print("sub.view(sub.size()[0], -1).size() : ", sub.view(sub.size()[0], -1).size()) #[16, 288]
            #print("self.bn_last_s: ", self.bn_last_s) # BatchNorm1d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #print("self.bn_last_s(sub.view(sub.size()[0], -1)): ", self.bn_last_s(sub.view(sub.size()[0], -1)))
            sub = self.linear_s(F.relu(self.bn_last_s(sub.view(sub.size()[0], -1)))) #@carina stacktrace points to this line to cause CUDA RunTimeError: Illegal memory 
            sub = sub.unsqueeze(2).repeat(1, 1, x.shape[1])

            bias = self.conv_bias(F.relu(self.bn_bias(x[:, :, -self.sim_size:].transpose(1, 2).contiguous()))).transpose(1,
                                                                                                                        2)
            # Compute action representation
            x = x[:, :, :-self.sim_size].transpose(1, 2).contiguous()
            x = self.conv1(F.relu(self.bn1(x)))
            x = self.conv2(F.relu(self.bn2(x)))
            x = self.conv3(F.relu(self.bn3(x)))

            # Compute Q(s,a)
            out = torch.cat([x, sub], dim=1)
            out = self.conv_final2(self.bn_final(out))
            return (torch.sigmoid(bias) * out.transpose(1, 2)).view(out.size()[0], -1) 
