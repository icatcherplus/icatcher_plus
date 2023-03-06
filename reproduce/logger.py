from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    class that writes training stats into tensorboard.
    to view when training on a remote target, you must launch tensorboard executable with no-hangup (nohup) on some port,
    then port forward into your local machine.
    """
    def __init__(self, opt):
        self.opt = opt
        if self.opt.rank == 0:
            self.writer = SummaryWriter(log_dir=opt.experiment_path)

    def write_scaler(self, category, name, scalar_value, iterations):
        if self.opt.rank == 0:
            final_name = category + "/" + name
            self.writer.add_scalar(final_name, scalar_value, iterations)

    def close(self):
        if self.opt.rank == 0:
            if self.writer is not None:
                self.writer.close()
