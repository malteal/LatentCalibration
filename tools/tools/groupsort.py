"Group Sort"
import torch as T
from torch import nn

from sklearn.datasets import make_moons, make_circles, make_classification



class GroupSort(nn.Module):
    # def __init__(self, n_groups: int, axis: int = -1):
    def __init__(self, previous_layer=None):
        super(GroupSort, self).__init__()
        # self.n_groups = n_groups
        # self.axis = axis
        self.previous_layer = previous_layer
        if self.previous_layer is None:
        # if len(self.previous_layer.weight.shape)==2:
            self._forward = self.linear
        elif len(self.previous_layer.weight.shape)==4:
            self.unfold_args = (self.previous_layer.kernel_size,
                                self.previous_layer.dilation,
                                self.previous_layer.padding,
                                self.previous_layer.stride
                                )
            self.fold_args = (self.previous_layer.kernel_size,
                            self.previous_layer.dilation,
                            self.previous_layer.padding,
                            self.previous_layer.stride
                            )
            self._forward = self.conv
    
    def linear(self, x: T.Tensor):
        if len(x.shape)==3:
            return x.sort(2)[0]
        else:
            return x.sort(1)[0]

    def conv(self, x: T.Tensor):
        x_unfold = nn.functional.unfold(x,*self.unfold_args)
        x_gs = self.linear(x_unfold)
        x_fold = nn.functional.fold(x_gs,x.shape[2:],*self.fold_args)
        return x_fold

    def forward(self, x: T.Tensor):
        return self._forward(x)
        # return group_sort(x, self.n_groups, self.axis)

    def extra_repr(self):
        return f"num_groups: {self.n_groups}"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    data, label = make_moons(50_000, noise=0.7)
    data = T.tensor(data).float()
    label = T.tensor(label)
    loss = nn.BCEWithLogitsLoss()

    n_var=1
    sp = T.nn.Softplus()
    activation = GroupSort()
    if True:
        inputs = T.meshgrid( T.linspace(-1,1, 5), T.linspace(-1,1, 5))
        inputs = T.concat([inputs[0][...,None], inputs[1][...,None]], -1).view(1,25,2)
        l_layer = nn.Linear(2, 4)
        output = activation(inputs)
        output = activation(l_layer(inputs))
        
    elif False:

        w = T.randn(n_var,3, requires_grad=True)
        w = w/T.max(T.tensor(1.0),w.abs().sum(axis=0))
        b = T.randn(1,3, requires_grad=True)
        w1 = T.randn(3,3, requires_grad=True)
        b1 = T.randn(1,3, requires_grad=True)
        inputs = T.linspace(-2,2, n_var*10, requires_grad=True).view(-1,n_var)
        
        model = nn.Sequential(nn.Linear(2, 1))
        # model.zero_grad()
        # linear_layer = activation(model(data))
        # output = loss(linear_layer, label.view(-1,1).float())
        # output.backward()
        # print(model[0].weight.grad)
        l_layer = nn.Linear(1, 3)
        l_layer.weight.data = l_layer.weight.sort(0)[0]

        fig, ax = plt.subplots(1,4, figsize=(16,6))
        if False:
            linear_layer = (inputs@w+b)#@w1+b1+inputs
        else:
            print(l_layer.weight)
            linear_layer = l_layer(inputs)
            
        for i in range(w.shape[-1]):
            ax[0].plot(np.ravel(inputs.detach().numpy()),  linear_layer[:, i].detach().numpy())

        l_layer.weight.data = l_layer.weight.sort(1)[0]
        print(l_layer.weight)
        linear_layer_gs = l_layer(inputs)
        # linear_layer_gs_nabla = T.autograd.grad(linear_layer_gs.mean(), inputs, retain_graph=True)
        # w_gs_nabla = T.autograd.grad(linear_layer_gs.mean(), w, retain_graph=True)

        linear_layer_gs_no_b = sp(linear_layer)
        # linear_layer_gs_no_b_nabla = T.autograd.grad(linear_layer_gs_no_b.mean(), inputs, retain_graph=True)
        # w_sp_nabla = T.autograd.grad(linear_layer_gs_no_b.mean(), w, retain_graph=True)

        linear_layer_sp = activation(sp(linear_layer))
        # linear_layer_sp_nabla = T.autograd.grad(linear_layer_sp.mean(), inputs, retain_graph=True)
        # w_gs_sp_nabla = T.autograd.grad(linear_layer_sp.mean(), w, retain_graph=True)

        for i in range(w.shape[-1]):
            ax[1].plot(np.ravel(inputs.detach().numpy()),
                        linear_layer_gs[:, i].detach().numpy())
            ax[2].plot(np.ravel(inputs.detach().numpy()),
                        linear_layer_gs_no_b[:, i].detach().numpy())
            ax[3].plot(np.ravel(inputs.detach().numpy()),
                        linear_layer_sp[:, i].detach().numpy())
    else: # conv
        n_var = 3
        inputs = T.randn(1,1,6,6)
        w = nn.Conv2d(1,1,3)
        output = w(inputs)
        activation = GroupSort(w)


        out_act_sp = sp(output)
        out_act = activation(output)