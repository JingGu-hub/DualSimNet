import argparse
import os
import numpy as np
from torch.nn.functional import mse_loss

from models.CNN import CNN
from models.Resnets import resnet18
from models.Transformer import Transformer
from utils.data_utils import build_dataset

import torch
import torch.nn.functional as F

from utils.double_cos import double_cos
from utils.features_prob_distilling import FeaturesProbKnowledgeDistilling
from utils.utils import create_file, set_seed, CustomMultiStepLR, create_dir, f1_scores, gmm_divide, gmm_divide_loss, \
    inter_intra_distance_compute, pairflip_penalty

from utils.constants import Multivariate2018_arff_DATASET as UEA_DATASET
from utils.constants import Four_dataset as OTHER_DATASET

import warnings
warnings.filterwarnings("ignore")

def pretrain_model(args, trainloader, model_path, dcos=None):
    out_dir = args.loss_save_dir + args.archive + '/' + str(args.dataset) + '/'
    loss_file = 'model_loss_%s_%s%.2f_seed%d.txt' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed)
    loss_file = create_file(out_dir, loss_file, 'epoch,train loss')

    # Build model conditionally
    # model = Transformer(input_dim=args.input_channel, input_length=args.seq_len, embedding_size=args.embedding_size, feature_size=args.features, num_classes=args.num_classes).cuda()
    model = CNN(input_channel=args.input_channel, num_classes=args.num_classes, series_length=args.seq_len, features=args.features).cuda()
    # model = resnet18(args, input_channel=args.input_channel, num_classes=args.num_classes, series_length=args.seq_len, features=args.features).cuda()
    model.train()

    # Optimizer and scheduler (Common setup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Training
    best_loss = float('inf')
    if not os.path.exists(model_path + '/' + args.res_model_name):
        for epoch in range(args.reconstructing_epochs):

            total_loss = 0
            for i, (inputs, targets, _, index) in enumerate(trainloader):
                inputs = inputs.cuda()

                # Model-specific forward
                restruction_input, _ = model(inputs, task_type='restruction')
                mse_loss = F.mse_loss(restruction_input, inputs, reduction='mean')

                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()

                total_loss = mse_loss.item() * inputs.shape[0]

            total_loss /= len(trainloader.dataset)
            print('epoch: %d, total_loss: %.4f' % (epoch, total_loss))
            with open(loss_file, "a") as myfile:
                myfile.write(str('Epoch:[%d/%d] train_loss:%.4f\n' % (epoch + 1, args.reconstructing_epochs, total_loss)))

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), model_path + args.res_model_name)

    model.load_state_dict(torch.load(model_path + args.res_model_name))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = CustomMultiStepLR(optimizer, milestones=[100, 150], gammas=[0.1, 0.1])

    features_all = torch.zeros((len(trainloader.dataset), args.features_num)).cuda()
    train_labels_all = torch.zeros(len(trainloader.dataset)).type(torch.LongTensor).cuda()
    correct, total, test_loss = 0, 0, 0
    for epoch in range(args.pretraining_epochs):
        scheduler.step()
        for i, (inputs, targets, _, index) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            model.train_freeze()

            optimizer.zero_grad()
            logits, features = model(inputs, task_type='classification')
            mean_loss = F.cross_entropy(logits, targets, reduction='mean')
            mean_loss.backward()
            optimizer.step()

            test_loss += mean_loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = torch.max(logits, 1)
            correct += predicted.eq(targets).cpu().sum().item()

            train_labels_all[index] = targets
            features_all[index] = features

        test_acc = 100. * correct / total
        test_loss /= total
        print('epoch: %d, total_loss: %.4f, train_acc: %.4f%%' % (epoch, test_loss, test_acc))
        with open(loss_file, "a") as myfile:
            myfile.write(str('Epoch:[%d/%d] train_loss:%.4f, train_acc: %.4f%%\n' % (epoch + 1, args.pretraining_epochs, test_loss, test_acc)))

        if dcos.is_update_key_sample(features_all, args.num_classes):
            torch.save(model.state_dict(), model_path + args.cls_model_name)
            dcos.init_key_sample(features_all, train_labels_all, None, args.num_classes, use_key_selection=args.use_key_selection)


def warmup_train(args, stu_model, teacher_model, dataloader, optimizer, tea_optimizer, num_classes=10, dcos=None):
    stu_model.train()
    teacher_model.train_freeze()

    train_loss = 0
    features_all = torch.zeros((len(dataloader.dataset), args.features_num)).cuda()
    train_labels_all = torch.zeros(len(dataloader.dataset)).type(torch.LongTensor).cuda()
    clean_labels_all = torch.zeros(len(dataloader.dataset)).type(torch.LongTensor).cuda()
    for i, (inputs, labels, clean_labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs, _ = stu_model(inputs, task_type='classification')
        loss = F.cross_entropy(outputs, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        with torch.no_grad():
            logits, features = teacher_model(inputs, task_type='classification')
        # tea_optimizer.zero_grad()
        # logits, features = teacher_model(inputs, task_type='classification')
        # ce_loss = F.cross_entropy(logits, labels, reduction='none')
        # mean_loss, clean_mask = gmm_divide_loss(ce_loss, loss_type=True)
        # mean_loss.backward()
        # tea_optimizer.step()

        clean_labels_all[index] = clean_labels.cuda()
        train_labels_all[index] = labels
        features_all[index] = features

    train_loss /= len(dataloader.dataset)
    if dcos.is_update_key_sample(features_all, num_classes):
        dcos.init_key_sample(features_all, train_labels_all, clean_labels_all, num_classes, use_key_selection=args.use_key_selection)

    return train_loss, dcos

def eval_train(args, eval_loader, stu_model, teacher_model, num_classes, dcos=None):
    stu_model.eval()
    teacher_model.eval()

    losses = torch.zeros(len(eval_loader.dataset)).cuda()
    outputs_all = torch.zeros((len(eval_loader.dataset), num_classes)).cuda()
    features_all = torch.zeros((len(eval_loader.dataset), args.features_num)).cuda()
    target_pred_all = torch.zeros((len(eval_loader.dataset))).type(torch.LongTensor).cuda()
    train_labels_all = torch.zeros(len(eval_loader.dataset)).type(torch.LongTensor).cuda()
    clean_labels_all = torch.zeros(len(eval_loader.dataset)).type(torch.LongTensor).cuda()

    total_ce_loss = 0
    for batch_idx, (inputs, targets, clean_labels, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs, _ = stu_model(inputs, task_type='classification')
            loss = F.cross_entropy(outputs, targets, reduction='none')

            losses[index] = loss
            train_labels_all[index] = targets
            clean_labels_all[index] = clean_labels.cuda()
            outputs_all[index] = outputs
            total_ce_loss += loss.mean().item() * inputs.shape[0]

            logits, features = teacher_model(inputs, task_type='classification')

        clean_labels_all[index] = clean_labels.cuda()
        train_labels_all[index] = targets

        features_all[index] = features
        target_pred = torch.argmax(logits, axis=1)
        target_pred_all[index] = target_pred

    # if dcos.is_update_key_sample(features_all, num_classes):
    #     dcos.init_key_sample(features_all, train_labels_all, clean_labels_all, num_classes, use_key_selection=args.use_key_selection)

    return losses, features_all, train_labels_all, clean_labels_all, outputs_all, dcos

def train(args, net, fp_kd, dataloader, optimizer, noise_scores, outputs_all, featrues_pred, mask=None):
    net.train()

    train_loss = 0
    for i, (inputs, labels, _, index) in enumerate(dataloader):
        inputs, labels, index = inputs.cuda(), labels.cuda(), index.cuda()
        mask_tensor = mask[index].cuda()
        original_unselected_index = (1 - mask[index]).nonzero().reshape(-1)
        unselected_index = index[original_unselected_index]

        outputs, _ = net(inputs, task_type='classification')
        loss = F.cross_entropy(outputs, labels, reduction='none')
        loss = torch.mean(mask_tensor * loss)

        if args.use_distill:
            # kd_loss = F.smooth_l1_loss(outputs[distill_index, :], noise_scores[distill_index, :])
            # rda_loss = fp_kd.compute_rda(noise_scores, outputs_all, outputs, index)

            if len(unselected_index) == 0:
                cur_loss, other_loss = 0, 0
            elif args.use_total_distill:
                cur_loss, other_loss = fp_kd.compute_rda(noise_scores, outputs_all, outputs, index)
            else:
                cur_loss, other_loss = fp_kd.compute_rda(noise_scores, outputs_all, outputs[original_unselected_index], unselected_index)

            loss = loss + args.distill_param * cur_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

    train_loss /= len(dataloader.dataset)

    return train_loss

def test(test_loader, model):
    model.eval()

    correct, total, test_loss = 0, 0, 0
    f1_macro, f1_weighted, f1_micro, test_num = 0, 0, 0, 0
    for batch_idx, (inputs, targets, index) in enumerate(test_loader):
        with torch.no_grad():
            test_num += 1
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs, task_type='classification')

            _, predicted = torch.max(outputs, 1)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()*targets.size(0)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            f1_mac, f1_w, f1_mic = f1_scores(outputs, targets)
            f1_macro += f1_mac
            f1_weighted += f1_w
            f1_micro += f1_mic

    test_acc = 100.*correct/total
    test_loss /= total
    f1_macro, f1_weighted, f1_micro = f1_macro / test_num, f1_weighted / test_num, f1_micro / test_num

    return test_acc, test_loss, f1_macro, f1_weighted, f1_micro

def main(archive='UEA', gpu_id=0, noise_type='symmetric', noise_rates=[0.5], deal_type='gmm', result_save_dir='./outputs/results/',
         use_key_selection=True, use_second_value=True, use_only_intra_class=False, sv_params=[1.0], use_distill=True, distill_params=[1.0], use_total_distill=False, ct_params=[0.]):
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, default='ctm', choices=['ctm', 'lstm', 'ff'], help='Model type to train.')
    parser.add_argument('-stu_model', type=str, default='Resnet', help='model type')  # 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'

    # CTM
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet18-4', help='Type of backbone featureiser.')
    parser.add_argument('--d_input', type=int, default=64, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads (CTM, LSTM).')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks (CTM, LSTM).')
    parser.add_argument('--positional_embedding_type', type=str, default='none', help='Type of positional embedding (CTM, LSTM).',
                        choices=['none', 'learnable-fourier', 'multi-learnable-fourier', 'custom-rotational'])
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).')
    parser.add_argument('--n_synch_out', type=int, default=64, help='Number of neurons to use for output synch (CTM only).')
    parser.add_argument('--n_synch_action', type=int, default=64, help='Number of neurons to use for observation/action synch (CTM only).')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory (CTM only).')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')

    # Hyparams setting
    parser.add_argument('--sv_param', type=float, default=0., help='Learning rate for the model.')
    parser.add_argument('--distill_param', type=float, default=0., help='Learning rate for the model.')
    parser.add_argument('--ct_param', type=float, default=0., help='Learning rate for the model.')

    # Training
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--model_lr', type=float, default=0.001, help='Learning rate for the model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--reconstructing_epochs', type=int, default=200, help='Number of training iterations.')
    parser.add_argument('--pretraining_epochs', type=int, default=30, help='Number of training iterations.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training iterations.')
    parser.add_argument('--warmup', type=int, default=30, help='Number of warmup steps.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay factor.')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--T', type=float, default=0.5, help='Learning rate for the model.')
    parser.add_argument('--class_i_rate', type=float, default=0.3, help='Learning rate for the model.')
    parser.add_argument('--features', type=int, default=64, help='features for training.')
    parser.add_argument('--embedding_size', type=int, default=256, help='features for training.')
    parser.add_argument('--features_num', type=int, default=128, help='features for training.')

    # Housekeeping
    parser.add_argument('--dataset', type=str, default='ArticularyWordRecognition', help='Dataset to use.')
    parser.add_argument('--data_root', type=str, default='../data/image_dataset/', help='Where to save dataset.')
    parser.add_argument('--data_dir', type=str, default='../data/Multivariate2018_arff/Multivariate_arff', help='dataset directory')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')

    parser.add_argument('--model_save_dir', type=str, default='./outputs/model_save/', help='model save directory')
    parser.add_argument('--result_save_dir', type=str, default='./outputs/result_save/', help='result save directory')
    parser.add_argument('--loss_save_dir', type=str, default='./outputs/loss_save/', help='result save directory')

    args = parser.parse_args()

    if archive == 'UEA':
        args.archive = archive
        datasets = UEA_DATASET
        args.data_dir = '../data/Multivariate2018_arff/Multivariate_arff'
    elif archive == 'other':
        args.archive = archive
        datasets = OTHER_DATASET
        args.data_dir = '../data/ts_noise_data/'
    torch.cuda.set_device(gpu_id)
    args.result_save_dir, args.use_key_selection, args.use_second_value, args.use_distill = result_save_dir, use_key_selection, use_second_value, use_distill
    args.use_total_distill, args.use_only_intra_class = use_total_distill, use_only_intra_class

    # seeds = [2026, 2027, 2028, 2029, 2030]
    # seeds = [512, 768, 1024, 2048, 4096]
    seeds = [2011, 2012, 2013, 2014, 2015]
    # seeds = [256, 128, 96, 64, 42]
    # seeds = [256]
    for dataset in datasets:
        args.dataset = dataset
        out_dir = args.result_save_dir + args.archive + '/' + str(args.dataset) + '/'
        total_file = create_file(out_dir, 'total.txt', 'statement,test_acc_list,test_acc,f1_macro_list,f1_macro,f1_weighted_list,f1_weighted,f1_micro_list,f1_micro', exist_create_flag=False)

        for noise_rate in noise_rates:
            args.noise_type = noise_type
            args.label_noise_rate = noise_rate

            for sv_param in sv_params:
                args.sv_param = sv_param
                for distill_param in distill_params:
                    args.distill_param = distill_param
                    for ct_param in ct_params:
                        args.ct_param = ct_param

                        test_acc_list, f1_macro_list, f1_weighted_list, f1_micro_list = [], [], [], []
                        for seed in seeds:
                            args.random_seed = seed
                            set_seed(args)

                            # Dataset
                            train_loader, test_loader, input_channel, seq_len, num_classes, clean_ids = build_dataset(args)
                            args.input_channel, args.seq_len, args.out_dims, args.num_classes = input_channel, seq_len, num_classes, num_classes
                            dcos = double_cos(args, len(train_loader.dataset))

                            # pretrain model if model parameters do not exist
                            model_path = args.model_save_dir + args.archive + '/' + str(args.dataset) + '/'
                            create_dir(model_path)
                            args.res_model_name = 'res_model_%s_%s%.2f.pth' % (args.dataset, args.noise_type, args.label_noise_rate)
                            args.cls_model_name = 'cls_model_%s_%s%.2f.pth' % (args.dataset, args.noise_type, args.label_noise_rate)
                            # args.model_name = 'model_%s_%s%.2f_seed%d.pth' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed)
                            if not os.path.exists(model_path + '/' + args.cls_model_name):
                                pretrain_model(args, train_loader, model_path, dcos)
 
                            # load pretrain model
                            # teacher_model = Transformer(input_dim=args.input_channel, input_length=args.seq_len, embedding_size=args.embedding_size, feature_size=args.features, num_classes=args.num_classes).cuda()
                            teacher_model = CNN(input_channel=args.input_channel, num_classes=args.num_classes, series_length=args.seq_len, features=args.features).cuda()
                            teacher_model.load_state_dict(torch.load(model_path + args.cls_model_name))
                            # stu_model = Transformer(input_dim=args.input_channel, input_length=args.seq_len, embedding_size=args.embedding_size, feature_size=args.features, num_classes=args.num_classes).cuda()
                            # args.features_num = args.input_channel * args.embedding_size
                            stu_model = CNN(input_channel=args.input_channel, num_classes=args.num_classes, series_length=args.seq_len, features=args.features).cuda()

                            # opt
                            tea_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.lr, eps=1e-8)
                            stu_optimizer = torch.optim.AdamW(stu_model.parameters(), lr=args.model_lr)
                            scheduler = CustomMultiStepLR(stu_optimizer, milestones=[100, 150], gammas=[0.1, 0.1])

                            fp_kd = FeaturesProbKnowledgeDistilling()

                            out_file = create_file(out_dir, '%s_%s_%s%.2f_seed%d_sv%.2f_distill%.2f_ct%.2f.txt' % (args.stu_model, args.dataset, args.noise_type, args.label_noise_rate, args.random_seed, args.sv_param, args.distill_param, args.ct_param),
                                                   'epoch,train_loss,test_loss,test_acc,f1_macro,f1_weighted,f1_micro')

                            last_five_accs, last_five_losses, last_five_f1_macro, last_five_f1_weighted, last_five_f1_micro = [], [], [], [], []
                            for epoch in range(args.epochs):
                                scheduler.step()

                                if epoch < args.warmup:
                                    # warmup train
                                    train_loss, dcos = warmup_train(args, stu_model, teacher_model, train_loader, stu_optimizer, tea_optimizer, num_classes, dcos=dcos)
                                else:
                                    input_loss1, features1, train_labels_all1, clean_labels_all, outputs_all, dcos = eval_train(args, train_loader, stu_model, teacher_model, num_classes, dcos=dcos)

                                    if deal_type == 'double_cos':
                                        scores1, noise_scores1, featrues_pred = dcos.compute_doubble_similarity(args, features1, train_labels_all1, clean_labels_all, num_classes, use_second_value=args.use_second_value)
                                        fp_kd.update_parameters(dcos.clas_use_index, featrues_pred)
                                        loss_type = False
                                    else:
                                        scores1, noise_scores1, featrues_pred = dcos.compute_doubble_similarity(args, features1, train_labels_all1, clean_labels_all, num_classes, use_second_value=args.use_second_value)
                                        fp_kd.update_parameters(dcos.clas_use_index, featrues_pred)
                                        scores1, noise_scores1, featrues_pred = input_loss1, noise_scores1, featrues_pred
                                        loss_type = True

                                    mask1 = gmm_divide(scores1, loss_type=loss_type)
                                    train_loss = train(args, stu_model, fp_kd, train_loader, stu_optimizer, noise_scores1, outputs_all, featrues_pred, mask=mask1)  # train net1

                                test_acc, test_loss, f1_macro, f1_weighted, f1_micro = test(test_loader, stu_model)
                                print('Epoch:[%d/%d], train_loss:%.4f, test_loss:%.4f, test_acc:%.4f, f1_macro:%.4f, f1_weighted:%.4f, f1_micro:%.4f' % (epoch + 1, args.epochs, train_loss, test_loss, test_acc, f1_macro, f1_weighted, f1_micro))
                                with open(out_file, "a") as myfile:
                                    myfile.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (epoch + 1, train_loss, test_loss, test_acc, f1_macro, f1_weighted, f1_micro))

                                if (epoch + 5) >= args.epochs:
                                    last_five_accs.append(test_acc)
                                    last_five_losses.append(test_loss)
                                    last_five_f1_macro.append(f1_macro)
                                    last_five_f1_weighted.append(f1_weighted)
                                    last_five_f1_micro.append(f1_micro)

                            # compute average result
                            test_accuracy = round(np.mean(last_five_accs), 4)
                            test_loss = round(np.mean(last_five_losses), 4)
                            f1_macro = round(np.mean(last_five_f1_macro), 4)
                            f1_weighted = round(np.mean(last_five_f1_weighted), 4)
                            f1_micro = round(np.mean(last_five_f1_micro), 4)
                            print('Test Accuracy:', test_accuracy, 'Test Loss:', test_loss, 'F1_macro:', f1_macro, 'F1_weighted:', f1_weighted, 'F1_micro:', f1_micro)

                            test_acc_list.append(test_accuracy.item())
                            f1_macro_list.append(f1_macro.item())
                            f1_weighted_list.append(f1_weighted.item())
                            f1_micro_list.append(f1_micro.item())

                        test_accuracy = round(np.mean(test_acc_list), 4)
                        mean_f1_macro, mean_f1_weighted, mean_f1_micro = round(np.mean(f1_macro_list), 4), round(np.mean(f1_weighted_list), 4), round(np.mean(f1_micro_list), 4)
                        with open(total_file, "a") as myfile:
                            myfile.write('%s_%s_%s%.2f_sv%.2f_distill%.2f_ct%.2f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f\n' % (args.stu_model, args.dataset, args.noise_type, args.label_noise_rate, args.sv_param, args.distill_param, args.ct_param,
                                         str(test_acc_list), test_accuracy, str(f1_macro_list), mean_f1_macro, str(f1_weighted_list), mean_f1_weighted, str(f1_micro_list), mean_f1_micro))


if __name__=='__main__':
    main(archive='UEA', gpu_id=3, noise_type='symmetric', noise_rates=[0.2], deal_type='double_cos', result_save_dir='./outputs/results/', use_second_value=True, use_distill=True, use_total_distill=False)
    # main(archive='UEA', gpu_id=3, noise_type='symmetric', noise_rates=[0.5], deal_type='double_cos', result_save_dir='./outputs/results/', use_second_value=True, use_distill=True, use_total_distill=False)
    # main(archive='UEA', gpu_id=3, noise_type='instance', noise_rates=[0.4], deal_type='double_cos', result_save_dir='./outputs/results/', use_second_value=True, use_distill=True, use_total_distill=False)
    # main(archive='UEA', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], deal_type='double_cos', result_save_dir='./outputs/results/', use_second_value=True, use_distill=True, use_total_distill=False)



