from config import parse_args
from network1 import Network

args = parse_args()
my_net = Network(args)
my_net.build()

if args.do_train :
    my_net.train()
#my_net.print_all_weights()