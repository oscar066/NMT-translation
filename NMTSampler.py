import os
import numpy as np
import torch
from nltk.translate import bleu_score
from training_utils import sentence_from_indices

chencherry = bleu_score.SmoothingFunction()

class NMTSampler:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        y_pred = self.model(x_source=batch_dict['x_source'],
                            x_source_lengths=batch_dict['x_source_lenghts'],
                            target_sequence=batch_dict['x_target'])
        self._last_batch['y_pred'] = y_pred

        attention_batched = np.stack(self.model.decoder._cache_p_attn).transpose
        self._last_batch['attention'] = attention_batched

    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)
    
    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().numpy()
        vocab = self.vectorizer.targer_vocab
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)
    
    def _get_sampled_sentence(self, index, return_string=True):
        _, all_indices = torch.max(self._last_batch['y_pred'], dim=2)
        sentence_indices = all_indices[index].cpu().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)
    
    def get_ith_item(self, index, return_string=True):
        output = {
            "source": self._get_source_sentence(index, return_string=return_string),
            "reference": self._get_reference_sentence(index, return_string=return_string),
            "sampled": self._get_sampled_sentence(index, return_string=return_string),
            "attention": self._last_batch['attention'][index]
        }

        reference = output['reference']
        hypothesis = output['hypothesis']

        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)

        output['bleu-4'] = bleu_score.sentence_bleu(reference=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry)
        
        return output