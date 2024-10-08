import torch 

import NMTEncoder as NMTEncoder
import NMTDecoder as NMTDecoder

class NMTModel(nn.Module):
    """
    A Neural Machine Translation Model
    """
    def __init__(self, source_vocab_size, source_embedding_size,
                 target_vocab_size, target_embedding_size, encoding_size,
                 target_bos_index):
        """
        Args:
            source_vocab_size (int) : number of unique words in source language
            source_embedding_size (int) : size of the source embeddings vectors
            target_vocab_size (int) : number of unique words in target language
            target_embedding_size (int) : size of the target embedding vectors
            encoding_size (int) : size of the encoder RNN
            target_bos_index (int) : index of BEGIN-OF-SEQUENCE token
        """
        super(NMTModel, self).__init__()
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        
        decoding_size = encoding_size * 2

        self.decoder = NMTDecoder(num_embeddings=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index)
        
        def foward(self, x_source, x_source_lengths, target_sequence):
            """
            The foward pass of the model
            Args:
                x_source (torch.Tensor) : the source text data tensor
                    x_source.shape should be (batch, vectorizer.max_source_length)
                x_source_lengths (torch.Tensors) : the length of the sequences in x_source
                target_sequence (torch.Tensor) : the target text data tensor
            Returns:
                decoded_states (tensor.Tensor) : Prediction vectors at each output step
            """
            encoder_state, final_hidden_states = self.encoder(x_source,
                                                              x_source_lengths)
            
            decoded_states = self.decoder(encoder_state=encoder_state,
                                          initial_hidden_state=final_hidden_states,
                                          target_sequence=target_sequence)
            
            return decoded_states