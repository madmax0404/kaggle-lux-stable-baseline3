from torch.optim.lr_scheduler import _LRScheduler

class GreedyLR(_LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=10, cooldown=0, warmup=0, 
                 min_lr=0, max_lr=10, smooth=False, window=5, reset=None):
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.warmup = warmup
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smooth = smooth
        self.window = window
        self.reset = reset
        
        self.best_loss = float('inf')
        self.warmup_counter = 0
        self.cooldown_counter = 0
        self.num_good_epochs = 0
        self.num_bad_epochs = 0
        self.loss_window = []
        
        super().__init__(optimizer)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics=None):
        if metrics is not None:
            current_lr = self.get_lr()[0]
            
            if self.smooth:
                self.loss_window.append(metrics)
                if len(self.loss_window) > self.window:
                    self.loss_window.pop(0)
                metrics = sum(self.loss_window) / len(self.loss_window)
            
            if metrics < self.best_loss:
                self.best_loss = metrics
                self.num_good_epochs += 1
                self.num_bad_epochs = 0
            else:
                self.num_good_epochs = 0
                self.num_bad_epochs += 1
            
            if self.warmup_counter < self.warmup:
                self.warmup_counter += 1
                new_lr = min(current_lr / self.factor, self.max_lr)
            elif self.cooldown_counter < self.cooldown:
                self.cooldown_counter += 1
                new_lr = max(current_lr * self.factor, self.min_lr)
            elif self.num_good_epochs >= self.patience:
                new_lr = min(current_lr / self.factor, self.max_lr)
                self.cooldown_counter = 0
            elif self.num_bad_epochs >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.warmup_counter = 0
            else:
                new_lr = current_lr
            
            new_lr = max(self.min_lr, min(new_lr, self.max_lr))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            if self.reset and self.last_epoch % self.reset == 0:
                self.best_loss = float('inf')
                self.warmup_counter = 0
                self.cooldown_counter = 0
                self.num_good_epochs = 0
                self.num_bad_epochs = 0
                self.loss_window = []
            
            self.last_epoch += 1
            return [new_lr]
        else:
            return self.get_lr()