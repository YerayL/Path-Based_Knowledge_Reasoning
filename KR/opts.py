""" Implementation of all available options """


def config_opts(parser):

    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')
    parser.add('--log_file', '-log_file', type=str, default=None,
              help="Output logs to a file under this path.")

    # train config
    parser.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")
    parser.add('--train_epochs', '-train_epochs', type=int, default=30)
    parser.add('--batch_size', '-batch_size', type=int, default=256)
    parser.add('--batch_first', '-batch_first', type=bool, default=True)
    parser.add('--weight_decay', '-weight_decay', type=float, default=0.00001)
    parser.add('--drop_out', '-drop_out', type=float, default=0)
    parser.add('--multi_gpu', '-multi_gpu', type=bool, default=True)
    parser.add("--device", type=str, default="cuda")
    parser.add("--root", type=str, default="data/data_output_csv/_aviation_airport_serves/")

