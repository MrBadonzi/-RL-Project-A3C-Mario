from torch.optim import Adam
from torch import zeros_like
from torch.optim import RMSprop


class GlobalAdam(Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = zeros_like(p.data) #exponentially decaying average of past gradients
                state['exp_avg_sq'] = zeros_like(p.data) #exponentially decaying average of past squared gradients to provide an adaptive learning rate
                state['exp_avg'].share_memory_() #moves the exp_avg from storage to the shared memory, thereafter to all agents in multiprocessing
                state['exp_avg_sq'].share_memory_()



class GlobalRMSprop(RMSprop):
    def __init__(self, params, lr):
        super(GlobalRMSprop, self).__init__(params, lr)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = zeros_like(p.data) #exponentially decaying average of past gradients
                state['square_avg'] =  zeros_like(p.data)
                state['exp_avg_sq'] = zeros_like(p.data) #exponentially decaying average of past squared gradients to provide an adaptive learning rate
                state['exp_avg'].share_memory_() #moves the exp_avg from storage to the shared memory, thereafter to all agents in multiprocessing
                state['exp_avg_sq'].share_memory_()
                state['square_avg'].share_memory_()