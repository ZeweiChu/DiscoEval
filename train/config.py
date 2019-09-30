import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='wiki')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--auto_disconnect', type="bool", default=True,
                             help='for slurm (default: True)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')
    basic_group.add_argument('--model',
                             type=str, default="basic",
                             choices=['basic', 'pos', 'quant_pos',
                                      'quant_attn_pos1', 'quant_attn_pos2',
                                      'quant_pos_reg'],
                             help='types of model')
    basic_group.add_argument('--uni_pred', type="bool", default=False,
                             help='only predict next sent (default: false)')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--eval_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--vocab_file', type=str, default=None,
                            help='vocabulary file')
    data_group.add_argument('--embed_file', type=str, default=None,
                            help='pretrained embedding file')

    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=1e-3,
                              help='learning rate')
    config_group.add_argument('-lrd', '--lr_decay_rate',
                              dest='lrd',
                              type=float,
                              default=0.,
                              help='learning rate decay rate')
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-4,
                              help='for avoiding numerical issues')
    config_group.add_argument('-edim', '--embed_dim',
                              dest='edim',
                              type=int, default=300,
                              help='size of embedding')
    config_group.add_argument('-dedim', '--dec_embed_dim',
                              dest='dedim',
                              type=int, default=300,
                              help='size of decoder input embedding')
    config_group.add_argument('-wd', '--word_dropout',
                              dest='wd',
                              type=float, default=0.0,
                              help='word dropout rate')
    config_group.add_argument('-dp', '--dropout',
                              dest='dp',
                              type=float, default=0.0,
                              help='dropout probability')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=None,
                              help='gradient clipping threshold')
    config_group.add_argument('-nb', '--n_bucket',
                              dest='nb',
                              type=int, default=20,
                              help='number of bucket')
    # recurrent neural network detail
    config_group.add_argument('-ensize', '--encoder_size',
                              dest='ensize',
                              type=int, default=300,
                              help='encoder hidden size')
    config_group.add_argument('-desize', '--decoder_size',
                              dest='desize',
                              type=int, default=300,
                              help='decoder hidden size')
    config_group.add_argument('-tw', '--tie_weight',
                              dest='tw', type="bool", default=False,
                              help='whether to tie embeddings')

    # feedforward neural network
    config_group.add_argument('-mhsize', '--mlp_hidden_size',
                              dest='mhsize',
                              type=int, default=300,
                              help='size of hidden size')
    config_group.add_argument('-mlplayer', '--mlp_n_layer',
                              dest='mlplayer',
                              type=int, default=2,
                              help='number of layer')

    # loss ratio
    config_group_show = parser.add_argument_group('model_configs_show')
    config_group_show.add_argument('-pratio', '--ploss_ratio',
                                   dest='pratio',
                                   type=float,
                                   default=1.0,
                                   help='ratio of paragraph position loss')
    config_group_show.add_argument('-sratio', '--sloss_ratio',
                                   dest='sratio',
                                   type=float,
                                   default=1.0,
                                   help='ratio of sentence position loss')
    config_group_show.add_argument('-lratio', '--logloss_ratio',
                                   dest='lratio',
                                   type=float,
                                   default=1.0,
                                   help='ratio of log loss')
    config_group_show.add_argument('-lvratio', '--level_loss_ratio',
                                   dest='lvratio',
                                   type=float,
                                   default=1.0,
                                   help='ratio of level prediction loss')
    config_group_show.add_argument('-dtratio', '--doc_title_loss_ratio',
                                   dest='dtratio',
                                   type=float,
                                   default=1.0,
                                   help='ratio of document title prediction loss')
    config_group_show.add_argument('-stratio', '--sec_title_loss_ratio',
                                   dest='stratio',
                                   type=float,
                                   default=1.0,
                                   help='ratio of section title prediction loss')

    # optimization
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')

    setup_group = parser.add_argument_group('train_setup')
    # train detail
    setup_group.add_argument('--save_dir', type=str, default=None,
                             help='model save path')
    setup_group.add_argument('--encoder_type',
                             type=str, default="lstm",
                             choices=['lstm', 'gru', 'wordavg', 'gru_attn'],
                             help='types of encoder')
    setup_group.add_argument('--decoder_type',
                             type=str, default="lstm",
                             choices=['lstm', "gru", "gru_cat",
                                      "bag_of_words"],
                             help='types of decoder')
    setup_group.add_argument('--n_epoch', type=int, default=1,
                             help='number of epochs')
    setup_group.add_argument('--batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--max_len', type=int, default=200,
                             help='max sentenc length')
    setup_group.add_argument('--opt', type=str, default='adam',
                             choices=['adam', 'sgd', 'rmsprop'],
                             help='types of optimizer: adam (default), \
                             sgd, rmsprop')
    setup_group.add_argument('--train_emb', type="bool", default=True,
                             help='whether to make embedding trainable')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=10,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=100,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--summarize', type="bool", default=False,
                            help='whether to summarize training stats\
                            (default: False)')
    return parser
