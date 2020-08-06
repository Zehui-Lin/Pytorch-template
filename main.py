import os
import time
import torch
import random
import argparse
import logging
import numpy as np
from tqdm import trange
from epoch_fun import train
from epoch_fun import val
from epoch_fun import test
from utils import check_dir
from utils import plot
from utils import savebest_weights
from dataset import MySet
from network import model
from torch.utils.data import DataLoader


def main(args):
    # 建立文件夹和文件
    output_path = "output_test/"+args.name
    txt_path = 'data/txt'+str(args.cv_index)
    result_path = os.path.join(output_path, "result")
    best_path = os.path.join(output_path, "best_weights")
    log_path = os.path.join(result_path, "Log.txt")
    check_dir(best_path)
    check_dir(output_path)
    check_dir(result_path)
    # 保存parser设置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler((os.path.join(result_path, 'Setting_Log.txt')), mode='w')
    logger.addHandler(fh)
    logger.info(args)
    # 加载数据
    train_set = MySet(txt_path, mode="train", is_debug=args.smoke_test)
    val_set = MySet(txt_path,  mode="val", is_debug=args.smoke_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=20 if not args.smoke_test else 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=20 if not args.smoke_test else 0)
    # 定义模型
    net = model().cuda()
    # 多GPU
    if len(args.gpu_id) > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method='tcp://localhost:8000', rank=0, world_size=1)
        net = torch.nn.parallel.DistributedDataParallel(net)

    # 损失以及优化器
    cost = torch.nn.CrossEntropyLoss()
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练结果记录
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    save = savebest_weights(args.num_model_to_save, best_path)
    t0 = time.time()
    # epoch循环
    for epoch in trange(args.num_epoch):
        train_loss, train_acc = train(net, train_loader, optimizer, cost)
        val_loss, val_acc = val(net, val_loader, cost)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        save.save_weight(net, val_acc, epoch)

        # 画出训练集和验证集的loss及acc
        plot(train_loss_list, 'train_loss', val_loss_list, 'val_loss',
             x_label="epoch", y_label="loss", title="Loss Curve-epoch", save_path=result_path)
        plot(train_acc_list, 'train_acc', val_acc_list, 'val_acc',
             x_label="epoch", y_label="acc", title="Acc Curve-epoch", save_path=result_path)

        info = [str(epoch).zfill(3), train_loss, val_acc]
        logtxt = open(log_path, "a")
        logtxt.write("Epoch: {} | Train Loss: {:.4f} Val ACC: {:.4f}\n".format(*info))
        print("\rEpoch: {} | Train Loss: {:.4f} Val ACC: {:.4f}".format(*info))
        logtxt.close()

    t2 = time.time()
    print("Optimization Finished!  Cost time:{:.1f} minutes".format((t2-t0)/60))
    print("Start test in best val model...")
    test_acc_list = []
    test_set = MySet(txt_path, mode="test", is_debug=args.smoke_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=20 if not args.smoke_test else 0)
    best_weight = os.listdir(best_path)
    for i in range(args.num_model_to_save):
        # 模型
        net.load_state_dict(torch.load(os.path.join(best_path, best_weight[i])))
        test_acc, tn, fp, fn, tp = test(net, test_loader)
        test_acc_list.append(test_acc)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
    tp = tp_list[np.argmax(test_acc_list)]
    tn = tn_list[np.argmax(test_acc_list)]
    fp = fp_list[np.argmax(test_acc_list)]
    fn = fn_list[np.argmax(test_acc_list)]
    precision = tp/(tp+fp)
    sensitive = tp/(tp+fn)
    specificity = tn/(tn+fp)
    F1score = 2*tp/(2*tp+fp+fn)
    logtxt = open(log_path, "a")
    logtxt.write("Test ACC: {} ,|sensitive:{:.2f}|specificity:{:.2f}|precision:{:.2f}|F1score:{:.2f}| the best: {:4f} and the weight name: {}\n".format(
        test_acc_list, np.max(test_acc_list), sensitive*100, specificity*100, precision*100, F1score*100, best_weight[np.argmax(test_acc_list)]))
    print("The test acc:{}, the best: {} and the weight name: {}\n".format(
        test_acc_list, np.max(test_acc_list), sensitive*100, specificity*100, precision*100, F1score*100, best_weight[np.argmax(test_acc_list)]))
    logtxt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 几乎每次需要改动的
    parser.add_argument('--name', type=str, help="folder name, must declare")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--smoke_test', action='store_true', help="finish fast to test")
    parser.add_argument('--cv_index', choices=('1', '2', '3', '4', ''), default='')
    # 不要轻易改动的
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, choices=('Adam', 'SGD'), default='Adam')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_model_to_save', type=int, default=5)
    args = parser.parse_args()
    assert args.name is not None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main(args)
