
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """
        Args:
            num_embeddings (int) : size of source vocabulary
            embedding_size (int) : size of the embedding vectors
            rnn_hidden_size (int): size of the RNN hidden state vectors
        """
        super(NMTEncoder, self).__init__()

        self.source_embedding = nn.Embedding(num_embeddings, embedding_size,
                                             padding_idx=0)
        
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True,
                            batch_first=True)
        
    def foward(self, x_source, x_lengths):
        """
        The foward pass of the model 
        Args:
            x_source (torch.Tensor) : the input data tensor
                x_source.shape is (batch, seq_size)
            x_lengths (torch.Tesors) : vector of lengths for each item in batch
        Returns:
            a tuple : x_unpacked (torch.Tensor), x_birnn (torch.Tensor)
                x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
                x_birnn_h.shape = (batch, rnn_hidden_size * 2)
        """
        x_embedded = self.source_embedding(x_source)
        # create PackedSequence; x_packed.data.shape=(number_items, embedding_size)

        x_lengths = x_lengths.detach().cpu().numpy()
        
        x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True)

        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        x_birnn_out , x_birnn_h = self.birnn(x_packed)

        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)

        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)

        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h