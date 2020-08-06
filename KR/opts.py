""" Implementation of all available options """


import argparse

def config_opts(parser):

    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               default="./config.json",
               help='config file save path')
    parser.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    parser.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--src_word_vec_size', '-src_word_vec_size',
              type=int, default=500,
              help='Word embedding size for src.')

def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. "
                   "Options are [text|img|audio|vec].")

def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add('--data', '-data', required=True,
              help='Path prefix to the ".train.pt" and '
                   '".valid.pt" file path from preprocess.py')

def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help="Path to model .pt file(s). "
                   "Multiple models can be specified, "
                   "for ensemble decoding.")
