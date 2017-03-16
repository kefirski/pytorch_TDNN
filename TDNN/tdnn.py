import torch as t
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self, kernels, input_embed_size, bias = False):
        super(TDNN, self).__init__()

        self.input_embed_size = input_embed_size

        self.kernels = [Parameter(t.Tensor(out_dim, input_embed_size, kW).normal_(0, 0.05))
                        for kW, out_dim in kernels]
        self._add_to_parameters(self.kernels, 'TDNN_kernel')

        self.use_bias = bias

        if self.use_bias:
            self.biases = [Parameter(t.Tensor(out_dim).normal_(0, 0.05))
                           for _, out_dim in kernels]
            self._add_to_parameters(self.biases, 'TDNN_biases')

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]

        :return: tensor with shape [batch_size, max_seq_len, depth_sum]

        applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """

        input_size = x.size()
        input_size_len = len(input_size)

        assert input_size_len == 4, \
            'Wrong input rang, must be equal to 4, but {} found'.format(input_size_len)

        [batch_size, seq_len, max_word_len, _] = input_size

        # leaps with shape
        x = x.view(-1, max_word_len, self.input_embed_size).transpose(1, 2).contiguous()

        xs = [F.relu(F.conv1d(x, kernel, bias=self.biases[i] if self.use_bias else None))
              for i, kernel in enumerate(self.kernels)]
        xs = [x.max(2)[0].squeeze(2) for x in xs]

        x = t.cat(xs, 1)
        x = x.view(batch_size, seq_len, -1)

        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)
