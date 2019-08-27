# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

from progress.bar import FillingSquaresBar
import torch
import quantlab.nets as nets


def train(logbook, net, device, loss_fn, opt, train_l):
    """Run one epoch of the training experiment."""
    meter.reset()
    bar = FillingSquaresBar('Training \t', max=len(trainloader))
    controllers = nets.Controller.getControllers(net)
    for i_batch, data in enumerate(trainloader):
        # load data onto device
        inputs, labels = data
        inputs         = inputs.to(device)
        labels         = labels.to(device)
        # fwdprop
        outputs        = net(inputs)
        loss           = loss_fn(outputs, labels)
        # update statistics
        logbook.meter.update(pr_outs, gt_labels, loss.item(), track_metric=logbook.track_metric)
        bar.suffix = 'Total: {total:} | ETA: {eta:} | Epoch: {epoch:4d} | ({batch:5d}/{num_batches:5d})'.format(
                total=bar.elapsed_td,
                eta=bar.eta_td,
                epoch=logbook.i_epoch,
                batch=i_batch + 1,
                num_batches=len(train_l))
        bar.suffix = bar.suffix + logbook.meter.bar()
        bar.next()
        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    for ctrlr in controllers:
        ctrlr.step(logbook.i_epoch, opt)
    bar.finish()
    stats = {
        'train_loss':   logbook.meter.avg_loss,
        'train_metric': logbook.meter.avg_metric
    }
    for k, v in stats.items():
        if v:
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
    logbook.writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], global_step=logbook.i_epoch)
    return stats


def test(logbook, net, device, loss_fn, test_l, valid=False):
    """Run a validation epoch."""
    logbook.meter.reset()
    bar_title = 'Validation \t' if valid else 'Test \t'
    bar       = FillingSquaresBar(bar_title, max=len(test_l))
    with torch.no_grad():
        for i_batch, data in enumerate(test_l):
            # load data onto device
            inputs, gt_labels     = data
            inputs                = inputs.to(device)
            gt_labels             = gt_labels.to(device)
            # forprop
            tensor_stats, pr_outs = net.forward_with_tensor_stats(inputs)
            loss                  = loss_fn(pr_outs, gt_labels)
            # update statistics
            logbook.meter.update(pr_outs, gt_labels, loss.item(), track_metric=True)
            bar.suffix = 'Total: {total:} | ETA: {eta:} | Epoch: {epoch:4d} | ({batch:5d}/{num_batches:5d})'.format(
                total=bar.elapsed_td,
                eta=bar.eta_td,
                epoch=logbook.i_epoch,
                batch=i_batch + 1,
                num_batches=len(test_l))
            bar.suffix = bar.suffix + logbook.meter.bar()
            bar.next()
    bar.finish()
    prefix = 'valid' if valid else 'test'
    stats = {
        prefix+'_loss':   logbook.meter.avg_loss,
        prefix+'_metric': logbook.meter.avg_metric
    }
    if valid:
        for k, v in stats.items():
            if v:
                logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
        for name, tensor in tensor_stats:
            logbook.writer.add_histogram(name, tensor, global_step=logbook.i_epoch)
    return stats
