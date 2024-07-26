import torch.nn as nn
import torch
from torchstat import stat
class PreMSNN(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=1):
        super(PreMSNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            #nn.ELU(),

            nn.MaxPool1d(8, stride=8),
            #nn.Dropout(0.3)
            )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            #nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            #nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            #nn.ELU(),
            #nn.Dropout(0.3)
        )



        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )
        self.dropout = nn.Dropout(0.2)


        # 加了一个线性层，全连接
        self.fc2 = nn.Linear(4096, 5)

        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch_size=120):
        out1 = self.module1(x)
        out2 = self.module2(x)

        out = torch.cat((out1, out2), dim=2)  ###拼接两个特征提取CNN网络的特征输出

        out = out.view(batch_size, -1)

        out = self.dropout(out)

        fc2_output = self.fc2(out)
        output = self.softmax1(fc2_output)

        return output


class MSNN(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(MSNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            #nn.ELU(),

            nn.MaxPool1d(8, stride=8),
            #nn.Dropout(0.3)
            )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            #nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            #nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

            #nn.ELU(),
            #nn.Dropout(0.3)
        )

        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16, stride=4, padding=0),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.3)
        )

        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )
        self.dropout = nn.Dropout(0.3)
        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(
                input_size=4096,
                hidden_size=gru_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,  ##双向GRU

        )

        # 加了一个线性层，全连接
        self.fc1 = nn.Linear(gru_hidden_size * 2, 200)

        # 加入了第二个全连接层
        self.fc3 = nn.Linear(200, 5)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x, batch_size=120, time_step=5):
        # print(x.shape)
        out1 = self.module1(x)
        out2 = self.module2(x)
        out = torch.cat((out1, out2), dim=2)  ###拼接两个特征提取CNN网络的特征输出
        out = out.view(batch_size, -1)

        # out = self.dropout(out)

        out = out.view(out.shape[0] // time_step, time_step, -1)
        # out = out.repeat(1, 5, 1)

        # out = out.view(out.shape[0], time_step, -1)
        g_out, h_n = self.gru(out, None)
        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        g_out = g_out.reshape(batch_size, -1)


        fc1_output = self.fc1(g_out)

        fc3_output = self.fc3(fc1_output)
        output = self.softmax2(fc3_output)
        return output


class viewMSNN(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(viewMSNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            # nn.ELU(),

            nn.MaxPool1d(8, stride=8),
            # nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            # nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )
    def forward(self, x, batch_size=120, time_step=5):
        out1 = self.block1(x)
        out2 = self.block3(x)
        return out1, out2



class LSTM(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(LSTM, self).__init__()


        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(
                input_size=3000,
                hidden_size=gru_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,  ##双向GRU

        )

        # 加了一个线性层，全连接
        self.fc1 = nn.Linear(gru_hidden_size * 2, 200)

        # 加入了第二个全连接层
        self.fc3 = nn.Linear(200, 5)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x, batch_size=120, time_step=5):

        out = x
        # out = out.repeat(1, 5, 1)

        out = out.view(out.shape[0] // 5, time_step, -1)
        g_out, h_n = self.gru(out, None)
        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        g_out = g_out.reshape(batch_size, -1)
        fc1_output = self.fc1(g_out)

        fc3_output = self.fc3(fc1_output)
        output = self.softmax2(fc3_output)
        return output

class viewLSTM(nn.Module):
    def __init__(self, batch_size=120, input_dim=3000, gru_hidden_size=512, out_size=300, num_layers=2):
        super(viewLSTM, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            # nn.ELU(),

            nn.MaxPool1d(8, stride=8),
            # nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(4, stride=4),
            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=12, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            # nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            # nn.ELU(),
            # nn.Dropout(0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),

            # nn.ELU(),
            # nn.Dropout(0.3)
        )


        self.module1 = nn.Sequential(
            self.block1,
            self.block2,
            self.block5,
            nn.MaxPool1d(4, stride=4),
        )

        self.module2 = nn.Sequential(
            self.block3,
            self.block4,
            self.block6,
            nn.MaxPool1d(2, stride=2),
        )
        self.dropout = nn.Dropout(0.2)
        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(
            input_size=4096,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  ##双向GRU

        )


    def forward(self, x, batch_size=120, time_step=5):
        out1 = self.module1(x)
        out2 = self.module2(x)
        out = torch.cat((out1, out2), dim=2)  ###拼接两个特征提取CNN网络的特征输出
        out = out.view(out.shape[0], -1)

        # out = self.dropout(out)

        out = out.view(out.shape[0] // time_step, time_step, -1)
        # out = out.repeat(1, 5, 1)

        # out = out.view(out.shape[0], time_step, -1)
        g_out, h_n = self.gru(out, None)
        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        g_out = g_out.reshape(batch_size, -1)
        return g_out


# myModel = MSNN()
# stat(myModel, (1,3000))
