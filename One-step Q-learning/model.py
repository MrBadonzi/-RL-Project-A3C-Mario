from torch import nn


class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.fc1 = nn.Linear(512, num_actions)
        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)


    def forward(self, x, hx, cx):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x=x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        return self.fc1(hx), hx, cx