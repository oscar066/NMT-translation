

class Vocabulary(object):
    """
    Class to process text and extract vocabulary for mapping
    """

    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict) : a pre-existing map of tokens
        """
        if token_to_idx is None:
            token_to_idx = {}
        
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token , idx in self._token_to_idx.items()}
        
    def to_serializable(self):
        """ returns a dictionary that can be serialized """

        return {'token_to_idx' : self._token_to_idx}
    
    @classmethod
    def from_serializable(cls, contents):
        """ Instantiate the Vocabulary from a serialized dictionary """
        return cls(**contents)
    
    def add_token(self, token):
        """
        Update mappings dict based on the token .

        Args:
            token (str) : the item to add into the Vocabulary
        Returns:
            index (int) : the interger corresponding to the token
        """

        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index
    
    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary
        
        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        """Retrieve the index associated with the token 
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]
    
    def lookup_index(self, index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._index_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]
    
    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    
    def __len__(self):
        return len(self._token_to_idx)