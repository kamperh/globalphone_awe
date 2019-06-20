"""
Iterators for passing in mini-batches.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from os import path
import numpy as np
import sys

sys.path.append(path.join("..", "src"))

from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE


#-----------------------------------------------------------------------------#
#                          BATCHING ITERATOR CLASSES                          #
#-----------------------------------------------------------------------------#

class SimpleIterator(object):
    """Iterator without bucketing."""
    
    def __init__(self, x_list, batch_size, shuffle_every_epoch=False):
        self.x_list = x_list
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.n_input = self.x_list[0].shape[-1]
        self.x_lengths = np.array([i.shape[0] for i in x_list])
        self.n_batches = int(len(self.x_lengths)/batch_size)
        self.indices = np.arange(len(self.x_lengths))
        np.random.shuffle(self.indices)
    
    def __iter__(self):

        if self.shuffle_every_epoch:
            np.random.shuffle(self.indices)
        
        for i_batch in range(self.n_batches):

            batch_indices = self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size
                ]
            
            batch_x_lengths = self.x_lengths[batch_indices]

            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.n_input),
                 dtype=NP_DTYPE
                )
            for i, length in enumerate(batch_x_lengths):
                seq = self.x_list[batch_indices[i]]
                batch_x_padded[i, :length, :] = seq

            yield (batch_x_padded, batch_x_lengths)


class SimpleBucketIterator(object):
    """An iterator with bucketing."""

    def __init__(self, x_list, batch_size, n_buckets,
            shuffle_every_epoch=False):
        self.x_list = x_list
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.n_input = self.x_list[0].shape[-1]
        self.x_lengths = np.array([i.shape[0] for i in x_list])
        self.n_batches = int(len(self.x_lengths)/batch_size)
        
        # Set up bucketing
        self.n_buckets = n_buckets
        sorted_indices = np.argsort([len(i) for i in x_list])
        bucket_size = int(len(self.x_lengths)/self.n_buckets)
        self.buckets = []
        for i_bucket in range(n_buckets):
            self.buckets.append(
                sorted_indices[i_bucket*bucket_size:(i_bucket + 1)*bucket_size]
                )
        self.shuffle()
            
    def shuffle(self):
        for i_bucket in range(self.n_buckets):
            np.random.shuffle(self.buckets[i_bucket])
        self.indices = np.concatenate(self.buckets)
    
    def __iter__(self):

        if self.shuffle_every_epoch:
            self.shuffle()
        
        for i_batch in range(self.n_batches):

            batch_indices = self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size
                ]
            
            batch_x_lengths = self.x_lengths[batch_indices]

            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.n_input),
                 dtype=NP_DTYPE
                )
            for i, length in enumerate(batch_x_lengths):
                seq = self.x_list[batch_indices[i]]
                batch_x_padded[i, :length, :] = seq

            yield (batch_x_padded, batch_x_lengths)


class PairedBucketIterator(object):
    """Iterator over bucketed pairs of sequences."""
    
    def __init__(self, x_list, pair_list, batch_size, n_buckets,
            shuffle_every_epoch=False, speaker_ids=None):

        # Attributes
        self.x_list = x_list
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.speaker_ids = speaker_ids

        self.n_input = self.x_list[0].shape[-1]
        self.x_lengths = np.array([i.shape[0] for i in x_list])
        # self.n_batches = int(len(self.x_lengths)/batch_size)
        self.n_batches = int(len(self.pair_list)/self.batch_size)
        
        # Set up bucketing
        self.n_buckets = n_buckets
        sorted_indices = np.argsort(
            [max(len(x_list[i]), len(x_list[j])) for i, j in pair_list]
            )
        # bucket_size = int(len(self.x_lengths)/self.n_buckets)
        bucket_size = int(len(self.pair_list)/self.n_buckets)
        self.buckets = []
        for i_bucket in range(n_buckets):
            self.buckets.append(
                sorted_indices[i_bucket*bucket_size:(i_bucket + 1)*bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for i_bucket in range(self.n_buckets):
            np.random.shuffle(self.buckets[i_bucket])
        self.indices = np.concatenate(self.buckets)
    
    def __iter__(self):

        if self.shuffle_every_epoch:
            self.shuffle()
        
        for i_batch in range(self.n_batches):
            
            batch_pair_list = [
                self.pair_list[i] for i in self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size]
                ]

            batch_indices_a = [i for i, j in batch_pair_list]
            batch_indices_b = [j for i, j in batch_pair_list]
            
            batch_lengths_a = self.x_lengths[batch_indices_a]
            batch_lengths_b = self.x_lengths[batch_indices_b]

            if self.speaker_ids is not None:
                batch_speaker_a = self.speaker_ids[batch_indices_a]
                batch_speaker_b = self.speaker_ids[batch_indices_b]
            
            n_pad = max(np.max(batch_lengths_a), np.max(batch_lengths_b))
            
            # Pad to maximum length in batch            
            batch_padded_a = np.zeros(
                (len(batch_indices_a), n_pad, self.n_input), dtype=NP_DTYPE
                )
            batch_padded_b = np.zeros(
                (len(batch_indices_b), n_pad, self.n_input), dtype=NP_DTYPE
                )
            for i, length in enumerate(batch_lengths_a):
                seq = self.x_list[batch_indices_a[i]]
                batch_padded_a[i, :length, :] = seq
            for i, length in enumerate(batch_lengths_b):
                seq = self.x_list[batch_indices_b[i]]
                batch_padded_b[i, :length, :] = seq
            
            if self.speaker_ids is None:
                yield (
                    batch_padded_a, batch_lengths_a, batch_padded_b,
                    batch_lengths_b
                    )
            else:
                yield (
                    batch_padded_a, batch_lengths_a, batch_padded_b,
                    batch_lengths_b, batch_speaker_b
                    )


class RandomSegmentsIterator(object):
    """An iterator that samples random subsequences for each batch."""
    
    def __init__(self, x_full_list, batch_size, n_buckets, min_dur=50,
            max_dur=100, shuffle_every_epoch=False, paired=False):
        self.x_full_list = x_full_list
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.shuffle_every_epoch = shuffle_every_epoch
        self.paired = paired
        self.n_input = self.x_full_list[0].shape[-1]
        self.n_batches = int(len(self.x_full_list)/batch_size)

        self.sample_segments()

    def sample_segments(self):
        self.x_list = []
        for cur_x in self.x_full_list:
            cur_len = cur_x.shape[0]
            dur = np.random.randint(self.min_dur, min(self.max_dur, cur_len))
            start = np.random.randint(0, cur_len - dur)
            self.x_list.append(cur_x[start:start + dur, :])
        self.x_lengths = np.array([i.shape[0] for i in self.x_list])
        sorted_indices = np.argsort([len(i) for i in self.x_list])
        bucket_size = int(len(self.x_lengths)/self.n_buckets)
        self.buckets = []
        for i_bucket in range(self.n_buckets):
            self.buckets.append(
                sorted_indices[i_bucket*bucket_size:(i_bucket + 1)*bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for i_bucket in range(self.n_buckets):
            np.random.shuffle(self.buckets[i_bucket])
        self.indices = np.concatenate(self.buckets)
        # blocks = [
        #     self.indices[i:i + self.batch_size] for i in range(0,
        #     len(self.indices), self.batch_size)
        #     ]
        # np.random.shuffle(blocks)
        # self.indices[:] = [b for bs in blocks for b in bs]
        
    def __iter__(self):

        if self.shuffle_every_epoch:
            self.sample_segments()
        
        for i_batch in range(self.n_batches):

            batch_indices = self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size
                ]
            
            batch_x_lengths = self.x_lengths[batch_indices]

            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.n_input),
                 dtype=NP_DTYPE
                )
            for i, length in enumerate(batch_x_lengths):
                seq = self.x_list[batch_indices[i]]
                batch_x_padded[i, :length, :] = seq

            if self.paired:
                yield (
                    batch_x_padded, batch_x_lengths, batch_x_padded,
                    batch_x_lengths
                    )
            else:
                yield (batch_x_padded, batch_x_lengths)


class LabelledBucketIterator(object):
    """Iterator with labels and bucketing."""
    
    def __init__(self, x_list, y, batch_size, n_buckets,
            shuffle_every_epoch=False):
        self.x_list = x_list
        self.y = y
        self.batch_size = int(np.floor(batch_size*0.5))  # batching is done
                                                         # over pairs, but
                                                         # target batch size
                                                         # given over items
                                                         # within pairs
        self.shuffle_every_epoch = shuffle_every_epoch
        self.n_input = self.x_list[0].shape[-1]
        self.x_lengths = np.array([i.shape[0] for i in x_list])
        self.pair_list = get_pair_list(y, both_directions=False)
        self.n_batches = int(len(self.pair_list)/self.batch_size)
        
        # Set up bucketing
        self.n_buckets = n_buckets
        sorted_indices = np.argsort(
            [max(len(x_list[i]), len(x_list[j])) for i, j in self.pair_list]
            )
        bucket_size = int(len(self.pair_list)/self.n_buckets)
        self.buckets = []
        for i_bucket in range(self.n_buckets):
            self.buckets.append(
                sorted_indices[i_bucket*bucket_size:(i_bucket + 1)*bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for i_bucket in range(self.n_buckets):
            np.random.shuffle(self.buckets[i_bucket])
        self.indices = np.concatenate(self.buckets)
    
    def __iter__(self):

        if self.shuffle_every_epoch:
            self.shuffle()
        
        for i_batch in range(self.n_batches):
            
            batch_pair_list = [
                self.pair_list[i] for i in self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size]
                ]

            batch_indices = list(
                set([i for i, j in batch_pair_list] + [j for i, j in
                batch_pair_list])
                )
            
            batch_x_lengths = self.x_lengths[batch_indices]
            batch_y = self.y[batch_indices]
            
            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.n_input),
                 dtype=NP_DTYPE
                )
            for i, length in enumerate(batch_x_lengths):
                seq = self.x_list[batch_indices[i]]
                batch_x_padded[i, :length, :] = seq

            yield (batch_x_padded, batch_x_lengths, batch_y)


class LabelledIterator(object):
    """
    Iterator without bucketing or padding but with labels.
    
    If `y_vec` is set to None, no labels are yielded.
    """
    
    def __init__(self, x_mat, y_vec, batch_size, shuffle_every_epoch=False):
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.n_batches = int(np.floor(x_mat.shape[0]/batch_size))
        self.indices = np.arange(x_mat.shape[0])
        self.shuffle()
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle_every_epoch:
            self.shuffle()
        for i_batch in range(self.n_batches):
            batch_indices = self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size
                ]
            if self.y_vec is None:
                yield (self.x_mat[batch_indices])
            else:
                yield (self.x_mat[batch_indices], self.y_vec[batch_indices])



#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def get_pair_list(labels, both_directions=True):
    """Return a list of tuples giving indices of matching types."""
    N = len(labels)
    match_list = []
    for n in range(N - 1):
        cur_label = labels[n]
        for cur_match_i in (n + 1 + np.where(np.asarray(labels[n + 1:]) ==
                cur_label)[0]):
            match_list.append((n, cur_match_i))
            if both_directions:
                match_list.append((cur_match_i, n))
    return match_list
