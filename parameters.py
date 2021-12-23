import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--eml', type=int, default=256, \
                        help='encoder_max_length')
    parser.add_argument('--dml', type=int, default=64, \
                        help='decoder_max_length')

    parser.add_argument('--save_model', type=str, default='model/', \
                        help='file save model train')
    
    parser.add_argument('--epochs', type=str, default=10, \
                        help='epochs train')

    parser.add_argument('--device', type=str, default='cuda:0', \
                        help='cuda or cpu')

    parser.add_argument('--checkpoint', type=str, default='checkpoint/checkpoint-40500', \
                        help='checkpoint train text sum')

    args = parser.parse_args()

    return args