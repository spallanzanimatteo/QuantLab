# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

from progress.bar import FillingSquaresBar
import torch
import quantlab.nets as nets


def train(logbook, meter, net, device, loss_fn, opt, trainloader):
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
        meter.update(outputs, labels, loss.item())
        bar.suffix = 'Total: {total:} | ETA: {eta:} | Epoch: {epoch:4d} | ({batch:5d}/{num_batches:5d})'.format(
                total=bar.elapsed_td,
                eta=bar.eta_td,
                epoch=logbook.i_epoch,
                batch=i_batch + 1,
                num_batches=len(trainloader))
        bar.suffix = bar.suffix + meter.bar()
        bar.next()
        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    for ctrlr in controllers:
        ctrlr.step(logbook.i_epoch, opt)
    bar.finish()
    stats = {
        'train_loss':   meter.avg_loss,
        'train_metric': meter.avg_metric
    }
    for k, v in stats.items():
        logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
    logbook.writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], global_step=logbook.i_epoch)
    return stats


def test(logbook, meter, net, device, loss_fn, testloader, valid=False):
    """Run a validation epoch."""
    meter.reset()
    bar_title = 'Validation \t' if valid else 'Test \t'
    bar       = FillingSquaresBar(bar_title, max=len(testloader))
    with torch.no_grad():
        for i_batch, data in enumerate(testloader):
            inputs, labels        = data
            inputs                = inputs.to(device)
            labels                = labels.to(device)
            outputs, tensor_stats = net.forward_with_tensor_stats(inputs)
            loss                  = loss_fn(outputs, labels)
            # update statistics
            meter.update(outputs, labels, loss.item())
            bar.suffix = 'Total: {total:} | ETA: {eta:} | Epoch: {epoch:4d} | ({batch:5d}/{num_batches:5d})'.format(
                total=bar.elapsed_td,
                eta=bar.eta_td,
                epoch=logbook.i_epoch,
                batch=i_batch + 1,
                num_batches=len(testloader))
            bar.suffix = bar.suffix + meter.bar()
            bar.next()
    bar.finish()
    prefix = 'valid' if valid else 'test'
    stats = {
        prefix+'_loss':   meter.avg_loss,
        prefix+'_metric': meter.avg_metric
    }
    if valid:
        for k, v in stats.items():
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
        for name, tensor in tensor_stats:
            logbook.writer.add_histogram(name, tensor, global_step=logbook.i_epoch)
    return stats
