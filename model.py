import torch
import numpy as np

class LSTM(torch.nn.Module):
    def __init__(self,data):
        super().__init__()

        self.hidden = 256
        self.layers = 2
        self.drop = 0.5
        self.data = data
        self.lstm = torch.nn.LSTM(len(self.data), self.hidden, self.layers, dropout=self.drop)

        self.dropout = torch.nn.Dropout(self.drop)

        self.fc = torch.nn.Linear(self.hidden, len(self.data))
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1,1)

    def forward(self, input, param):

        x1, (hidden, cell) = self.lstm(input, param)
        x2 = self.dropout(x1)
        x3 = x2.view(x2.size()[0]*x2.size()[1], self.hidden)
        x4 = self.fc(x3)

        return x4, hidden, cell

    def init_hidden(self, seq_len):
        weight = next(self.parameters()).data
        hidden = weight.new(self.layers, seq_len, self.hidden).zero()
        cell = weight.new(self.layers, seq_len, self.hidden).zero()
        return hidden, cell

def data2OneHot(data, num_output):

    # One hot encode data
    init = np.zeros((np.multiply(*data.shape), num_output), dtype=np.float32)
    init[np.arange(init.shape[0]), data.flatten()] = 1
    init = init.reshape((*data.shape, num_output))

    return init

def gen_batches(data, batch_size, seq_size):
    pass

def train(data, model, cuda):
    num_epochs = 200
    learning_rate = 0.001
    len_chunk = 100
    seq_len = 10
    step_len = 50

    unique_notes = tuple(set(data))
    num_unique = len(unique_notes)
    int2note = dict(enumerate(unique_notes))
    note2int = {val: key for key, val in int2note.items()}
    encoded_data = np.array([note2int[note] for note in data])


    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    validation_amount = 0.2
    validation_split = int(len(encoded_data) * (1-validation_amount))
    train_data, validation_data = encoded_data[:validation_split], encoded_data[validation_split:]

    print_every = 5

    if cuda:
        model.cuda()

    for epoch in range (num_epochs):
        hidden,cell = model.init_hidden(seq_len)

        for input, target in gen_batches(encoded_data, seq_len, step_len):
            input = data2OneHot(input,num_unique)
            input = torch.from_numpy(input)
            target = torch.from_numpy(input)

            model.zero_grad()

            if cuda:
                input = input.cuda()
                target = target.cuda()

            output, hidden, cell = model.forward(input, (hidden, cell))

            if cuda:
                loss = criterion(output, target.view(seq_len*step_len).type(torch.cuda.LongTensor))
            else:
                loss = criterion(output, target.view(seq_len * step_len).type(torch.LongTensor))

            loss.backward()

            # Clip to reduce exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

        if epoch % print_every == 0:
            v_hidden, v_cell = model.init_hidden(seq_len)
            v_losses = []

            for input, target in gen_batches(validation_data, seq_len, step_len):
                input = data2OneHot(input, num_unique)
                input = torch.from_numpy(input)
                target = torch.from_numpy(input)

                if cuda:
                    input = input.cuda()
                    target = target.cuda()

                output, (v_hidden, v_cell) = model.forward(input, (v_hidden, v_cell))

                if cuda:
                    v_loss = criterion(output, target.view(seq_len*step_len).type(torch.cuda.LongTensor))
                else:
                    v_loss = criterion(output, target.view(seq_len * step_len).type(torch.LongTensor))

                v_losses.append(v_loss.item())

            print("Epoch: {}, loss: {}, v_loss_mean:{}".format(epoch, loss.item(), np.mean(v_losses)))


if __name__ == "__main__":
    print("Starting")
    data = [] #Make a call to matteo's stuff

