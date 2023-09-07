import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self,input_size, embed_dim, hidden_dim, num_class):

        super().__init__()
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        torch.backends.cudnn.enabled = False

        self.embedding = nn.EmbeddingBag(input_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, text, offsets):
        #----- DEBUG------
        # print(f"Text:{text}")
        # print(f"Text Size: {text.size()}")
        embedded = self.embedding(text, offsets)

        #----- DEBUG------
        # print(f"Embedded: {embedded}")
        # print(f"Embedded Size: {embedded.size()}")

        embedded.unsqueeze(1)
        embedded = torch.reshape(embedded, (embedded.size(0),1,100)).to(self.device)
        

        h0 = torch.zeros( embedded.size(0), self.hidden_dim) 
        output, hidden = self.rnn(embedded)

        #----- DEBUG------
        # print(f"Output: {output}")
        # print(f"Output size: {output.size()}")
        # print(f"Hidden: {hidden}")
        # print(f"Hidden Size: {hidden.size()}")

        # print(f"Hidden Squeeze: {hidden.squeeze(0)}")
        # print(hidden.squeeze(0).size())
        # print(f"FC: {self.fc(hidden.squeeze(0))}")
        # print(self.fc(hidden.squeeze(0)).size())

        assert torch.equal(output[:, -1, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))
    
    

