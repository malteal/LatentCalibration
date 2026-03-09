"Create the PICNN model"
import numpy as np
import torch as T
import torch.nn as nn
from tqdm import tqdm

from . import layers
from ..utils import transformations as trans

class PICNN(nn.Module):
    "Create the PICNN model"

    def __init__(
        self,
        cvx_dim: int,
        cvx_layersize: int,
        cvx_act: str,
        n_layers: int,
        noncvx_dim: int = 0,
        noncvx_layersize: int = 0,
        noncvx_act: str = "",
        act_params: dict = None,
        act_enforce_cvx: str = "softplus",
        act_weight_zz: str = "softplus",
        correction_trainable: bool = False,
        device: str = "cpu",
        verbose: bool = True,
        logit: bool = False,
        aug_icnn: bool = False,
        **kwargs,
    ) -> None:
        # T.set_default_dtype(T.float64)
        """Initializing the PICNN model

        Parameters
        ----------
        noncvx_act : str
            Name of the desired activation function
        cvx_act : str
            Name of the desired activation function
        noncvx_layersize : list
            list of number of layers and size
        cvx_layersize : list
            list of number of layers and size
        norm : str, optional
            to activate batch norm or not, by default False
        device : str, optional
            which device to run on, by default "cpu"
        """

        super().__init__()
        self.n_layers = n_layers
        self.cvx_dim = cvx_dim
        self.noncvx_dim = noncvx_dim

        # construct the layer sizes
        self.noncvx_layersize = [noncvx_layersize] * self.n_layers
        self.cvx_layersize = [cvx_layersize] * self.n_layers
        self.cvx_layersize.append(1)
        self.cvx_layersize.insert(0, self.cvx_dim)
        self.noncvx_layersize.insert(0, self.noncvx_dim)
        self.total_layer_len = len(self.cvx_layersize) - 1

        self.device = device
        self.act_weight_zz = act_weight_zz
        self.correction_trainable = correction_trainable
        self.cvx_act = cvx_act
        self.act_enforce_cvx = act_enforce_cvx
        self.noncvx_act = noncvx_act
        self.PICNN_bool = self.noncvx_dim > 0
        self.scaler = kwargs.get("scaler", None)
        self.logit = logit
        self.use_norm = kwargs.get("use_norm", False)

        # augment icnn from cpflows https://arxiv.org/abs/2012.05942
        self.aug_icnn=aug_icnn

        # symmetric activation function for the first layer
        self.sym_act=kwargs.get('sym_act', True)
        
        # activation function for the output correction
        self.g_corr_str=kwargs.get('g_corr', 'softplus')
        
        # activate onnx export if the network has to be able to be exported
        self.onnx_export=kwargs.get('onnx_export', False)

        # activation function force convexity on the linear transformations
        self.act_params = {} if act_params is None else act_params

        # build network
        self.get_network()

        if verbose:
            n_trainable = self.count_trainable_parameters()
            print(f"Number of trainable parameters: {int(n_trainable)}")

        # push network to device
        self.to(self.device)

    def count_trainable_parameters(self):
        sum_trainable = np.sum(
            [i.numel() for i in self.parameters() if i.requires_grad]
        )
        return sum_trainable

    def get_network(self):
        """initialize all parameters"""
        # more-or-less following the nomenclature from
        # arXiv:1609.07152

        # =============================================================================
        # init activation functions
        # =============================================================================

        self.act_enforce_cvx = (
            layers.get_act_funcs(  # TODO should this be a standalone class
                self.act_enforce_cvx, self.act_params, device=self.device
            )
        )

        # activation functions for non cvx layers
        self.gtilde_act = layers.get_act_funcs(
            self.noncvx_act, self.act_params, device=self.device
        )

        # activation function for output corrections
        if self.correction_trainable:
            self.g_corr = layers.get_act_funcs(self.g_corr_str, {}, device=self.device)
        

        # activation functions for cvx layers
        self.g_act = layers.get_act_funcs(
            self.cvx_act, self.act_params, device=self.device
        )
        
        # symmetric activation function for the first layer
        # this is allowed to be non-convex
        if self.sym_act:
            self.g_sym = layers.get_act_funcs("symsoftplus", {}, device=self.device)
        else:
            self.g_sym = self.g_act

        self.noncvx_layersize = (
            [0] * self.total_layer_len
            if self.noncvx_layersize is None
            else self.noncvx_layersize
        )

        # shorthand:
        zsize = self.cvx_layersize
        usize = self.noncvx_layersize
        ysize = zsize[0]
        # =============================================================================
        # Build network
        # =============================================================================
        # create trainable parameters
        self.layer_zz = nn.ModuleList([])
        self.layer_y = nn.ModuleList([])
        
        if self.aug_icnn:
            self.layer_aug = nn.ModuleList([])

        # convex layers
        if zsize[-1] != 1:
            raise ValueError(
                "Last layer of the convex part has to be 1 - change convex_layersizes"
                " to 1 as last"
            )

        # ICNN init
        for lay in range(self.total_layer_len):
            z_output = zsize[lay + 1] if (not self.aug_icnn or zsize[lay + 1]==1) else zsize[lay + 1] // 2
            
            self.layer_zz.append(
                layers.PositiveLayer(
                    zsize[lay],
                    z_output,
                    act_func=self.act_weight_zz,
                    device=self.device,
                    deactivate=lay == 0,
                )
            )
            self.layer_y.append(layers.Linear(ysize, z_output, bias=True))
            
            if self.aug_icnn:
                # augment icnn from cpflows
                self.layer_aug.append(layers.Linear(ysize, z_output, bias=True))

        # Additional layer to make it a PICNN
        if self.PICNN_bool:
            # create trainable parameters
            self.layer_zu = nn.ModuleList([])
            self.layer_yu = nn.ModuleList([])
            self.layer_u = nn.ModuleList([])
            self.layer_uutilde = nn.ModuleList([])

            for lay in range(self.total_layer_len):
                self.layer_yu.append(layers.Linear(usize[lay], ysize))
                self.layer_zu.append(layers.Linear(usize[lay], zsize[lay]))
                self.layer_u.append(layers.Linear(usize[lay], zsize[lay + 1], bias=False))

            for lay in range(self.total_layer_len - 1):
                self.layer_uutilde.append(layers.Linear(usize[lay], usize[lay + 1]))

            # the first z_i input is zero and it shouldnt be used in training
            self.layer_u[0].requires_grad = False

            # trainable output layer
            self.layer_uutilde.append(layers.Linear(usize[-1], 2))
        elif self.correction_trainable:
            # trainable output layer
            if self.g_corr_str == "softplus":
                self.w0 = nn.parameter.Parameter(
                    T.log(T.exp(T.ones(1)) - 1).to(device=self.device)
                )
                self.w1 = nn.parameter.Parameter(
                    T.log(T.exp(T.ones(1)/2) - 1).to(device=self.device)
                    # T.zeros(1).to(device=self.device)
                )
            else:
                # usually for sigmoid
                self.w0 = nn.parameter.Parameter(
                    T.tensor(2.0).float().to(device=self.device)
                    )
                self.w1 = nn.parameter.Parameter(
                    T.tensor(-2.0).float().to(device=self.device)
                )
            
        # create iterative norm layer
        if self.use_norm:
            self.cvx_norm = layers.IterativeNormLayer((1,self.cvx_dim), max_iters=10_000)
            if self.PICNN_bool:
                self.noncvx_norm = layers.IterativeNormLayer((1, self.noncvx_dim), max_iters=10_000)


    def get_trainable_parameters(self):  # pylint: disable=arguments-differ
        """yiels all parameters

        Yields
        ------
        nn.parameter.Parameter
            return iter of the trainable parameters
        """
        for parameter in self.parameters():
            if (
                isinstance(parameter, (nn.parameter.Parameter, T.Tensor))
                and parameter.requires_grad
            ):
                yield parameter
    
    def forward(self, cvx_input: T.Tensor, noncvx_input: T.Tensor = None) -> T.Tensor:
        """Run the forward data flow of the model

        Parameters
        ----------
        xs_input : T.Tensor
            conditional distribution
        ys_input : T.Tensor
            source distribution

        Returns
        -------
        T.Tensor
            return the output of the network h(theta,x)
        """
        if self.onnx_export:
            return self.transport(cvx_input, noncvx_input)
        else:
            return self._forward(cvx_input, noncvx_input)

    def _forward(self, cvx_input: T.Tensor, noncvx_input: T.Tensor = None) -> T.Tensor:
        # self.eval()
        """Run the forward data flow of the model

        Parameters
        ----------
        xs_input : T.Tensor
            conditional distribution
        ys_input : T.Tensor
            source distribution

        Returns
        -------
        T.Tensor
            return the output of the network h(theta,x)
        """
        
        
        # run norms
        cvx_input_or = cvx_input.clone()

        if self.use_norm:
            cvx_input = self.cvx_norm(cvx_input)

            if self.PICNN_bool:
                noncvx_input = self.noncvx_norm(noncvx_input)
        if self.aug_icnn:
            zi_value = self.AUGFICNN(cvx_input)
        elif self.PICNN_bool:
            zi_value, ui_value = self.PICNN(cvx_input, noncvx_input)
        else:
            zi_value = self.FICNN(cvx_input)

        if self.correction_trainable & self.PICNN_bool:
            # using softplus to make it positive
            ui_value = self.g_corr(ui_value)

            # trainable out ratio
            zi_value = layers.concat_and_sum(
                ui_value[:, :1] / 2 * T.sum(cvx_input_or * cvx_input_or, axis=1, keepdim=True),
                ui_value[:, 1:] * zi_value
            )
        elif hasattr(self, "w0") & self.correction_trainable:
            zi_value = layers.concat_and_sum(self.g_corr(self.w0) / 2
                        * T.sum(cvx_input_or * cvx_input_or, axis=1, keepdim=True),
                        self.g_corr(self.w1) * zi_value
            )
        # else:
        #     zi_value = layers.concat_and_sum(1/2 * T.sum(cvx_input_or * cvx_input_or, axis=1, keepdim=True),
        #                                       zi_value)
        return zi_value

    def AUGFICNN(self, cvx_input: T.Tensor) -> T.Tensor:
        # self.eval()
        """Run the forward data flow of the model

        Parameters
        ----------
        cvx_input : T.Tensor
            source distribution

        Returns
        -------
        T.Tensor
            return the output of the network h(theta,x)
        """

        zi_value = T.zeros_like(cvx_input)

        for i in range(self.total_layer_len-1):
            z = self.layer_zz[i](zi_value)
            x = self.layer_y[i](cvx_input)
            
            # augment the input
            aug = self.layer_aug[i](cvx_input)
            aug = self.g_sym(aug) if self.sym_act else self.g_act(aug)

            zi_value = self.g_sym(z + x) if i==0 else self.g_act(z + x)
                
            # concat input with augmented input
            zi_value = T.cat([zi_value, aug], 1)
        
        return self.layer_zz[-1](zi_value)+self.layer_y[-1](cvx_input)

    def FICNN(self, cvx_input: T.Tensor) -> T.Tensor:
        # self.eval()
        """Run the forward data flow of the model

        Parameters
        ----------
        cvx_input : T.Tensor
            source distribution

        Returns
        -------
        T.Tensor
            return the output of the network h(theta,x)
        """

        zi_value = T.zeros_like(cvx_input)

        for i in range(self.total_layer_len):

            z = self.layer_zz[i](zi_value)
            
            y = self.layer_y[i](cvx_input)
            
            if i == self.total_layer_len-1:
                zi_value = layers.concat_and_sum(z, y)
            elif i==0:
                zi_value = self.g_sym(
                layers.concat_and_sum(z, y)
                )
            else:
                zi_value = self.g_act(
                    layers.concat_and_sum(z, y)
                )
        return zi_value

    def PICNN(self, cvx_input: T.Tensor, noncvx_input: T.Tensor) -> T.Tensor:
        # self.eval()
        """Run the forward data flow of the model

        Parameters
        ----------
        cvx_input : T.Tensor
            conditional distribution
        noncvx_input : T.Tensor
            source distribution

        Returns
        -------
        T.Tensor
            return the output of the network h(theta,x)
        """
        ui_value = noncvx_input

        zi_value = T.zeros_like(cvx_input, requires_grad=True)
        # zi_value = T.zeros((cvx_input.shape[0], 1), device=self.device)
        

        for i in range(self.total_layer_len):
            yterm = cvx_input * self.layer_yu[i](ui_value)

            yterm = self.layer_y[i](yterm)

            zterm = zi_value * self.act_enforce_cvx(self.layer_zu[i](ui_value))

            zi_value = self.g_act(
                layers.concat_and_sum(
                self.layer_zz[i](zterm), yterm, self.layer_u[i](ui_value)
            ))

            ui_value = self.gtilde_act(self.layer_uutilde[i](ui_value))

        return zi_value, ui_value

    def transport(self, totransport, conditionals=None, create_graph=True):
        if not totransport.requires_grad:
            totransport.requires_grad = True

        cvx_output = self._forward(totransport, conditionals)

        output = T.autograd.grad(
            outputs=cvx_output,
            inputs=totransport,
            retain_graph=create_graph,
            create_graph=create_graph,
            grad_outputs=T.ones_like(cvx_output),
            allow_unused=False
        )[0]

        # add in bkg points
        if self.logit:
            output = trans.logit(trans.probsfromlogits(output))

        return output, cvx_output

    def chunk_transport(
        self,
        totransport,
        conditionals=None,
        sig_mask=None,
        n_chunks=20,
        pbar_bool: bool = False,
    ) -> T.Tensor:
        "transport large datasamples in chunks - only for eval"
        if sig_mask is None:
            sig_mask = T.ones(len(totransport)) == 1
            
        if conditionals is None:
            conditionals = T.zeros_like(totransport[:, :1])
            conditionals = conditionals.to(self.device)

        transport = []
        for totrans, conds, mask in tqdm(
            zip(
                totransport.chunk(n_chunks),
                conditionals.chunk(n_chunks),
                sig_mask.chunk(n_chunks),
            ),
            total=n_chunks,
            disable=not pbar_bool,
            leave=False,
        ):
            # clone because inplace operations not allow
            _transport = totrans.clone()

            # do transport
            _transport[mask] = (
                self.transport(
                    totrans.to(self.device)[mask], conds.to(self.device)[mask], create_graph=False
                )[0]
                .cpu()
                .detach()
            )

            # append to lst
            transport.append(_transport.cpu().detach())

        transport = T.concat(transport, 0)

        return transport


class nICNN(nn.Module):
    def __init__(self, config, n_layers=3, **kwargs):
        super().__init__()
        self.config = config
        self.n_layers = n_layers
        self.PICNN = nn.ModuleList([config(**kwargs) for _ in range(self.n_layers)])
        self.layer_zz = self.PICNN[0].layer_zz
        self.cvx_layersize = self.PICNN[0].cvx_layersize
        self.noncvx_layersize = self.PICNN[0].noncvx_layersize
        self.cvx_dim = self.PICNN[0].cvx_dim
        self.noncvx_dim = self.PICNN[0].noncvx_dim


    def forward(self, x, conds=None):

        for picnn in self.PICNN[:-1]:
            x, cvx = picnn.transport(x)

        return cvx

    def transport(self, totransport, conditionals=None, create_graph=True):
        for picnn in self.PICNN:
            totransport, cvx = picnn.transport(
                totransport, conditionals, create_graph=create_graph
            )
        return totransport, cvx

    def chunk_transport(
        self,
        totransport,
        conditionals=None,
        sig_mask=None,
        n_chunks=20,
        pbar_bool: bool = False,
    ):
        for picnn in self.PICNN:
            totransport = picnn.chunk_transport(
                totransport,
                conditionals,
                sig_mask=sig_mask,
                n_chunks=n_chunks,
                pbar_bool=pbar_bool,
            )
        return totransport
