

class NMTVectorizer(object):
    """
    The Vectorizer which coordinates the Vocabularies and puts them to use
    """
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        """
        Args:
            source_vocab (SequenceVocabulary) : maps source words to intergers
            target_vocab (SequenceVocabulary) : maps target words to intergers
            max_source_length (int) : the longest sequence in the source dataset
            max_target_length (int) : the longest sequence in the target dataset
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @classmethod
    def from_dataframe(cls, bitext_df):
        """
        Instantiate the vectorizer from the dataset dataframe

        Args:
            bitext_df (pandas.DataFrame) : the parallel text dataframe

        Returns:
            an instance of the NMTVectorizer
        """

        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length , max_target_length = 0, 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_tokens.add_token(token)

            return cls(source_vocab, target_vocab, max_source_length, max_target_length)