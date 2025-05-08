import torch

from brf_snn.functional import StepDoubleGaussianGrad
################################################################
# Neuron update functional
################################################################

DEFAULT_MASK_PROB = 0

TRAIN_B_OFFSET = True
DEFAULT_RF_B_OFFSET = 1.0

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_B_OFFSET_A = 1
DEFAULT_RF_ADAPTIVE_B_OFFSET_B = 6

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.0

DEFAULT_RF_ADAPTIVE_OMEGA_A = 10
DEFAULT_RF_ADAPTIVE_OMEGA_B = 50

DEFAULT_RF_THETA = 1 # .99

# Reset: Keep (1 - Zeta) of the membrane potential
# start with constant initialization
TRAIN_ZETA = False
DEFAULT_RF_ZETA = 0.00

DEFAULT_RF_ADAPTIVE_ZETA_A = 0
DEFAULT_RF_ADAPTIVE_ZETA_B = 0

TRAIN_DT = False
DEFAULT_DT = 0.01
DEFAULT_RF_ADAPTIVE_DT = 0.01


def hrf_update(
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (complex value)
        v: torch.Tensor,
        ref_period: torch.Tensor,
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # torch.Tensor 0.01
        theta: float = DEFAULT_RF_THETA,
):

    # damped oscillatory activity dim = (1, hidden_size)
    # membrane update u dim = (batch_size, hidden_size)
    v = v + u.mul(dt)
    u = u + x.mul(dt)-b.mul(u).mul(2*dt)-torch.square(omega).mul(v).mul(dt)


    # generate spike
    z = StepDoubleGaussianGrad.apply(u - theta - ref_period)
    ref_period = ref_period.mul(0.9) + z

    # reset membrane potential
    # u = u.mul(1 - z.mul(theta).mul(zeta))
    # v = v.mul(1 - z.mul(theta).mul(zeta))
    return z, u, v, ref_period


################################################################
# Layer classes
################################################################
class HRFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            mask_prob: float = DEFAULT_MASK_PROB,
            b_offset: float = DEFAULT_RF_B_OFFSET,
            adaptive_b_offset: bool = TRAIN_B_OFFSET,
            adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_OFFSET_A,
            adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_OFFSET_B,
            omega: float = DEFAULT_RF_OMEGA,
            adaptive_omega: bool = TRAIN_OMEGA,
            adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_A,
            adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_B,
            dt: float = DEFAULT_DT,
            bias: bool = False
    ) -> None:
        super(HRFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        # LinearMask: applies mask only to hidden recurrent weights in forward pass
        # linear.weight initialized with xavier_uniform_
        # self.mask_prob = mask_prob
        #
        # self.linear = rf.LinearMask(
        #     in_features=input_size,
        #     out_features=layer_size,
        #     bias=bias,
        #     mask_prob=mask_prob,
        #     lbd=input_size - layer_size,
        #     ubd=input_size,
        # )

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        omega = omega * torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)


        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        self.dt = dt

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, v, ref_period = state

        in_sum = self.linear(x)

        omega = torch.abs(self.omega)

        b_offset = torch.abs(self.b_offset)

        b = omega.square().mul(0.005) + b_offset + ref_period

        z, u, v, ref_period = hrf_update(
            x=in_sum,
            u=u,
            v=v,
            ref_period=ref_period,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, ref_period
