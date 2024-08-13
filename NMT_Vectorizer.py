import numpy as np
import pandas as pd

class NMT_Vectorizer(object):
    """
    The Vectorizer which coordinates the vocabularies and puts them to use
    """
    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """
        Vectorize the provided indices
        Args:
            indices (list) : a list of intergers that represent a sequence
            vector_length (int) : forces the length of the index vector
            mask_index (int) : the mask_index to use; almost always 0
        """
        if vector_length < 0:
            vector_length = len(indices)
        
        vector = np.zeros(vector_length, dytpe=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        return vector

    def _get_source_indices(self, text):
        """
        Return the vectorizer source text

        Args:
            text (str) : the source text; tokens separated by space
        Returns:
            indices (list) : list of intergers representing the text
        """ 
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.lookup_token(token)
                       for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)
        return indices 
    

    def _get_target_indices(self, text):
        """
        Return the vectorized source text
        Args:
            text (str) : the source text; tokens should be separated by spaces
        Returns:
            a tuple: (x_indices, y_indices)
                x_indices (list) : list of ints; observations in target decoder
                y_indices (list) : list of ints; observations in target decoder
        """
        indices = [self.target_vocab.lookup_token(token)
                   for token in text.split(" ")]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]

        return x_indices, y_indices
    
    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        """
        Returns the vectorized source and target text
        Args: 
            source_text (str) : text from the source language
            target_text (str) : text from the target language
            use_dataset_max_lengths (bool) : Whether to use the max vector lengths

        Returns:
            The vectorized data point as a dictionary with keys:
                source_vector, target_x_vector, target_y_vector, source_length
        """
        source_vector_length = - 1
        target_vector_length = - 1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices,
                                        vector_length=source_vector_length,
                                        mask_index=self.source_vocab.mask_index)
        
        target_x_indices , target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self.vectorize(target_x_indices,
                                         vector_length=target_vector_length,
                                         mask_index=self.target_vocab.mask_index)
        
        target_y_vector = self.vectorize(target_y_indices,
                                         vector_length=target_vector_length,
                                         mask_index = self.target_vocab.mask_index)
        
        return {
            "source_vector" : source_vector,
            "target_x_vector" : target_x_vector,
            "target_y_vector" : target_y_vector,
            "source_length" : len(source_indices)
        }

