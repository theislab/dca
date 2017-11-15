import numpy as np
from scipy.special import digamma

import torch
from tqdm import trange
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Function, Variable

_epsilon = 1e-10

def check_dicts(inputs, outputs):
    inlens = np.unique(np.array([len(x) for x in inputs.values()]))
    assert inlens.size == 1, 'Not all iterables are of equal length'

    outlens = np.unique(np.array([len(x) for x in outputs.values()]))
    assert outlens.size == 1, 'Not all iterables are of equal length'

    assert inlens[0] == outlens[0], 'Input output length do not match'


class DictTensorDataset(Dataset):
    def __init__(self, inputs, outputs):
        check_dicts(inputs, outputs)
        inlens = np.unique(np.array([len(x) for x in inputs.values()]))

        self.inputs =  {k: torch.from_numpy(v)
                        if type(v).__name__ == 'ndarray' else v for k, v in inputs.items()}

        self.outputs = {k: torch.from_numpy(v)
                        if type(v).__name__ == 'ndarray' else v for k, v in outputs.items()}
        self.length = inlens[0]

    def __getitem__(self, index):
        return ({k: v[index] for k, v in self.inputs.items()},
                {k: v[index] for k, v in self.outputs.items()})

    def __len__(self):
        return self.length

    def type(self, dtype):
        return DictTensorDataset({k: v.type(dtype) for k, v in self.inputs.items()},
                                 {k: v.type(dtype) for k, v in self.outputs.items()})

    def cuda(self):
        return DictTensorDataset({k: v.cuda() for k, v in self.inputs.items()},
                                 {k: v.cuda() for k, v in self.outputs.items()})


class ExpModule(torch.nn.Module):
    def __init__(self, eps=1e6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x).clamp(max=self.eps)


class EarlyStopping(object):
    def __init__(self, patience=10, metric='val_loss', threshold=1e-4, verbose=0):
        assert patience > 0, 'Patience must be positive'
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.last_epoch = 0
        self.num_bad_epochs = 0
        self.verbose = verbose
        self.metric = metric

    def step(self, metrics):
        assert self.metric in metrics, 'Metric undefined'
        self.last_epoch += 1
        current = metrics[self.metric][-1]

        if self.best is None:
            self.best = current
            return True

        if  current < self.best:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            if self.verbose:
                print('Epoch %s: early stopping' % self.last_epoch)
            return False

        return True


class Lgamma(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return torch.lgamma(input)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors

        res = Variable(torch.from_numpy(digamma(input.numpy())).type_as(input))
        return grad_output*res
lgamma = Lgamma.apply

 # log gamma code from pyro:
 # https://github.com/uber/pyro/blob/dev/pyro/distributions/util.py
def lgamma2(xx):
    gamma_coeff = (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    )
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = torch.ones_like(x) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def lgamma3(z):
    gamma_r10 = 10.900511

    pi = torch.zeros_like(z)
    pi.fill_(np.pi)
    gamma_c = 2.0 * torch.sqrt(np.e/pi)

    #if z < 0:
    #  return torch.log(pi) - torch.log(torch.abs(torch.sin(pi*z))) - lgamma(1.0 - z)

    sum = 2.48574089138753565546e-5
    sum += 1.05142378581721974210  / z
    sum += -3.45687097222016235469 / (z + 1.0)
    sum += 4.51227709466894823700  / (z + 2.0)
    sum += -2.98285225323576655721 / (z + 3.0)
    sum += 1.05639711577126713077  / (z + 4.0)
    sum += -1.95428773191645869583e-1 / (z + 5.0)
    sum += 1.70970543404441224307e-2  / (z + 6.0)
    sum += -5.71926117404305781283e-4 / (z + 7.0)
    sum += 4.63399473359905636708e-6  / (z + 8.0)
    sum += -2.71994908488607703910e-9 / (z + 9.0)

    # For z >= 0 gamma function is positive, no abs() required.
    return torch.log(gamma_c) + (z - 0.5)*torch.log(z  + gamma_r10 - 0.5) - (z - 0.5) + torch.log(sum)


class NBLoss(torch.nn.Module):
    def __init__(self, theta_shape=None, theta_dtype=torch.Tensor):
        super().__init__()

        if theta_shape is not None:
            theta = torch.Tensor(*theta_shape).log_normal_(0.1, 0.01).type(theta_dtype)
            self.register_parameter('theta', torch.nn.Parameter(theta))

    def forward(self, input, target, theta=None):
        eps = _epsilon

        if theta is None:
            theta = 1.0/(torch.exp(self.theta).clamp(max=1e7)+eps)

        t1 = -lgamma(target+theta+eps)
        t2 = lgamma(theta+eps)
        t3 = lgamma(target+1.0)
        t4 = -(theta * (torch.log(theta+eps)))
        t5 = -(target * (torch.log(input+eps)))
        t6 = (theta+target) * torch.log(theta+input+eps)

        res = t1+t2+t3+t4+t5+t6
        return res.mean()


class ZINBLoss(torch.nn.Module):
    def __init__(self, theta_shape=None, pi_ridge=0.0):
        super().__init__()
        self.pi_ridge = pi_ridge

        if theta_shape is not None:
            theta = torch.Tensor(*theta_shape).log_normal_(0.1, 0.01)
            self.register_parameter('theta', torch.nn.Parameter(theta))

    def forward(self, mean, pi, target, theta=None):
        eps = _epsilon

        if theta is None:
            theta = 1.0/(torch.exp(self.theta).clamp(max=1e6)+eps)

        # reuse existing NB nll
        nb_case = self.nb(mean, target, theta) - torch.log(1.0-pi+eps)

        zero_nb = torch.pow(theta / (theta + mean + eps), theta)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb) + eps)

        zero_mask = (target == 0.0).type_as(zero_case)
        nb_mask = 1.0 - zero_mask

        result = zero_mask*zero_case + nb_mask*nb_case

        if self.pi_ridge:
            ridge = self.pi_ridge*tf.square(pi)
            result += ridge

        return result.mean()

    def nb(self, input, target, theta):
        eps = _epsilon

        t1 = -lgamma(target+theta+eps)
        t2 = lgamma(theta+eps)
        t3 = lgamma(target+1.0)
        t4 = -(theta * (torch.log(theta+eps)))
        t5 = -(target * (torch.log(input+eps)))
        t6 = (theta+target) * torch.log(theta+input+eps)

        res = t1+t2+t3+t4+t5+t6
        return res


class ZINBEMLoss(torch.nn.Module):
    def __init__(self, theta_shape=None, pi_ridge=0.0):
        super().__init__()
        self.pi_ridge = pi_ridge

        if theta_shape is not None:
            theta = torch.Tensor(*theta_shape).log_normal_(0.1, 0.01)
            self.register_parameter('theta', torch.nn.Parameter(theta))

    def forward(self, mean, pi, target, zero_memberships, theta=None):
        eps = _epsilon

        if theta is None:
            theta = 1.0/(torch.exp(self.theta).clamp(max=1e6)+eps)

        nb_memberships = 1.0 - zero_memberships

        zero_comp = pi.clone()
        zero_comp.masked_fill_(target != 0.0, 0.0)
        zero_comp = zero_memberships * torch.log(zero_comp+eps)

        nb_comp = torch.log(1.0-pi+eps) - self.nb(mean, target, theta)
        nb_comp = nb_memberships * nb_comp

        result = -(zero_comp + nb_comp)

        if self.pi_ridge:
            ridge = self.pi_ridge*tf.square(pi)
            result += ridge

        return result.mean()

    def nb(self, input, target, theta):
        eps = _epsilon

        t1 = -lgamma(target+theta+eps)
        t2 = lgamma(theta+eps)
        t3 = lgamma(target+1.0)
        t4 = -(theta * (torch.log(theta+eps)))
        t5 = -(target * (torch.log(input+eps)))
        t6 = (theta+target) * torch.log(theta+input+eps)

        res = t1+t2+t3+t4+t5+t6
        return res

    def zero_memberships(self, mean, pi, target, theta=None):
        eps = _epsilon

        if theta is None:
            theta = 1.0/(torch.exp(self.theta).clamp(max=1e6)+eps)

        nb_zero_ll = torch.pow((theta/(theta+mean+eps)), theta)

        memberships = pi.clone()
        memberships = memberships / (memberships+((1.0-pi)*nb_zero_ll)+eps)
        memberships.masked_fill_(target != 0.0, 0.0)

        return memberships


def train(model_dict, loss_dict, model, loss, optimizer, epochs=1,
          val_split=0.1, val_data=None, batch_size=32, grad_clip=5.0,
          shuffle=True, verbose=0, early_stopping=None, scheduler=None,
          dtype='float'):

    check_dicts(model_dict, loss_dict)
    dataset = DictTensorDataset(model_dict, loss_dict)

    if dtype == 'cuda':
        dataset = dataset.cuda()
        model = model.cuda()
        loss = loss.cuda()
    elif dtype == 'double':
        dataset = dataset.type('torch.DoubleTensor')
        model = model.double()
        loss = loss.double()
    elif dtype == 'float':
        dataset = dataset.type('torch.FloatTensor')
        model = model.float()
        loss = loss.float()
    else:
        raise 'Unknown dtype'

    if shuffle:
        # shuffle dataset
        idx = torch.randperm(len(dataset))
        dataset = DictTensorDataset(*dataset[idx])

    if val_data is not None:
        train_data = dataset
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    elif val_split > 0.:
        off = int(len(dataset)*(1.0-val_split))
        train_data = DictTensorDataset(*dataset[:off])
        val_data = DictTensorDataset(*dataset[off:])
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    else:
        train_data, val_data = dataset, None

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    result = {'loss': [], 'model': model, 'early_stop': False}

    if verbose:
        it = trange(epochs)
    else:
        it = range(epochs)

    for epoch in it:
        train_batch_losses = []

        for modeld, lossd in loader:
            cur_batch_size = len(lossd['target'])
            modeld = {k: Variable(v) for k, v in modeld.items()}
            lossd  = {k: Variable(v) for k, v in lossd.items()}

            def closure():
                optimizer.zero_grad()
                pred = model(**modeld)
                if not isinstance(pred, dict): pred = {'input': pred}
                l = loss(**pred, **lossd)
                train_batch_losses.append(l.data.numpy()[0]*(cur_batch_size/batch_size))
                l.backward()
                if grad_clip:
                    for pg in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm(pg['params'], grad_clip)
                return l

            optimizer.step(closure)

        result['loss'].append(np.array(train_batch_losses).mean())

        if val_data:
            for modeld, lossd in val_loader:
                modeld = {k: Variable(v) for k, v in modeld.items()}
                lossd  = {k: Variable(v) for k, v in lossd.items()}

                pred = model(**modeld)
                if not isinstance(pred, dict): pred = {'input': pred}
                l = loss(**pred, **lossd)
                result.setdefault('val_loss', []).append(l.data.numpy()[0])

        if verbose:
            text = 'Epoch: %s training loss: %s val loss: %s' % ((epoch+1), result['loss'][-1],
                    result['val_loss'][-1] if 'val_loss' in result else  '---')
            it.set_description(text)

        if scheduler:
            if val_data is not None:
                scheduler.step(result['val_loss'][-1])
            else:
                if epoch == 0 and scheduler.verbose:
                    print('Validation data not specified, using training loss for lr scheduling')
                scheduler.step(result['loss'][-1])

        if early_stopping and not early_stopping.step(result):
            result['early_stop'] = True
            return result

    return result



def train_em(model_dict, loss_dict, model, loss,
             optimizer, epochs=1, m_epochs=1, val_split=0.1, grad_clip=5.0,
             batch_size=32, shuffle=True, verbose=0, early_stopping=None,
             scheduler=None, dtype='float'):

    memberships = torch.from_numpy(np.zeros_like(loss_dict['target']))
    loss_dict['zero_memberships'] = memberships
    check_dicts(model_dict, loss_dict)
    dataset = DictTensorDataset(model_dict, loss_dict)

    if dtype == 'cuda':
        dataset = dataset.cuda()
        model = model.cuda()
        loss = loss.cuda()
    elif dtype == 'double':
        dataset = dataset.type('torch.DoubleTensor')
        model = model.double()
        loss = loss.double()
    elif dtype == 'float':
        dataset = dataset.type('torch.FloatTensor')
        model = model.float()
        loss = loss.float()
    else:
        raise 'Unknown dtype'

    if shuffle:
        idx = torch.randperm(len(dataset))
        dataset = DictTensorDataset(*dataset[idx])

    if val_split > 0.:
        off = int(len(dataset)*(1.0-val_split))
        train_data = DictTensorDataset(*dataset[:off])
        val_data = DictTensorDataset(*dataset[off:])
    else:
        train_data, val_data = dataset, None

    ret = {'loss': []}
    if verbose:
        it = trange(int(np.ceil(epochs/m_epochs)))
    else:
        it = range(int(np.ceil(epochs/m_epochs)))

    for i in it:
        train_ret = train(model_dict=train_data.inputs, loss_dict=train_data.outputs,
                          model=model, loss=loss, optimizer=optimizer,
                          epochs=m_epochs, shuffle=shuffle, verbose=0,
                          batch_size=batch_size, grad_clip=grad_clip, val_data=val_data,
                          val_split=0.0, early_stopping=early_stopping,
                          dtype=dtype)
        ret['loss'] += train_ret['loss']

        if val_data:
            ret.setdefault('val_loss', []).extend(train_ret['val_loss'])

        if verbose:
            text = 'Epoch: %s training loss: %s val loss: %s' % ((i+1), ret['loss'][-1], ret['val_loss'][-1] if 'val_loss' in ret else  '---')
            it.set_description(text)

        pred = model(**{k: Variable(v) for k, v in train_data.inputs.items()}) #we need variables here
        memberships = loss.zero_memberships(**pred, target=Variable(train_data.outputs['target'])).data
        train_data.outputs['zero_memberships'] = memberships.clone()

        if val_data is not None:
            pred = model(**{k: Variable(v) for k, v in val_data.inputs.items()})
            memberships = loss.zero_memberships(**pred,
                                                target=Variable(val_data.outputs['target'])).data
            val_data.outputs['zero_memberships'] = memberships.clone()

        if scheduler:
            if val_data is not None:
                scheduler.step(ret['val_loss'][-1])
            else:
                if epoch == 0 and scheduler.verbose:
                    print('Validation data not specified, using training loss for lr scheduling')
                scheduler.step(ret['loss'][-1])

        if train_ret['early_stop']:
            break

    pred = model(**{k: Variable(v) for k, v in dataset.inputs.items()})
    ret['memberships'] = loss.zero_memberships(**pred,
                                               target=Variable(dataset.outputs['target'])).data.numpy()
    ret['model'] = model

    return ret

