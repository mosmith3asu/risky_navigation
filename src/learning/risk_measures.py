import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

####################################################################
### Risk-Sensitivity ###############################################
####################################################################


class CumulativeProspectTheory:
    def __init__(self, b, lam, eta_p, eta_n, delta_p, delta_n,
                 compiled=True, offset_ref=False):
        """
        Instantiates a CPT object that can be used to model human risk-sensitivity.
        :param b: reference point determining if outcome is gain or loss
        :param lam: loss-aversion parameter
        :param eta_p: exponential gain on positive outcomes
        :param eta_n: exponential loss on negative outcomes
        :param delta_p: probability weighting for positive outcomes
        :param delta_n: probability weighting for negative outcomes
        """
        # assert b==0, "Reference point must be 0"
        assert isinstance(b,(int,float)) or b=='mean', "b must be int, float, or 'mean'"
        self.mean_value_ref = (b=='mean')
        self.offset_ref = offset_ref

        self.b = b if not self.mean_value_ref else None
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

        self.compiled_expectation = compiled

        self._f_vmap = None

        self.is_rational = True
        if self.b != 0:
            self.is_rational = False
        elif self.lam != 1:
            self.is_rational = False
        elif self.eta_p != 1:
            self.is_rational = False
        elif self.eta_n != 1:
            self.is_rational = False
        elif self.delta_p != 1:
            self.is_rational = False
        elif self.delta_n != 1:
            self.is_rational = False


    def sample_expectation_batch(self, X, *args,**kwargs ):
        """Estimates CPT-value from only samples (no probs) in batch form"""
        if isinstance(X, torch.Tensor):
            return self._pt_sample_expectation_batch(X, *args,**kwargs)
        elif isinstance(X, np.ndarray):
            return self._np_sample_expectation_batch(X, *args,**kwargs)
        else:  raise ValueError("X must be a torch.Tensor, np.ndarray, or list.")

    def _np_sample_expectation_batch(self, X):
        """Estimates CPT-value from only samples (no probs) in batch form (NumPy version).

        X: shape (B, N) value samples.
        Returns: shape (B,) CPT value per batch element.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a np.ndarray, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array of value samples (B,N), got {X.shape}")

        B, N = X.shape
        if N < 1:
            raise ValueError("X must have at least one sample per batch element (N >= 1).")

        # Preserve original dtype for return (mimics rho.to(dtype=_dtype))
        orig_dtype = X.dtype

        # Compute in float64 for numerical stability
        X64 = np.ascontiguousarray(X.astype(np.float64, copy=False))

        # Reference point b
        if getattr(self, "mean_value_ref", False):
            b = X64.mean(axis=1, keepdims=True)  # (B, 1)
            self.b = b
        else:
            b = getattr(self, "b", None)
            if b is None:
                raise ValueError("self.b is None. Provide a reference point b or enable mean_value_ref.")

            # Normalize b to shape (B,1) or scalar-broadcastable
            if np.isscalar(b) or isinstance(b, (np.number,)):
                b = float(b)  # scalar
            elif isinstance(b, np.ndarray):
                if b.ndim == 0:
                    b = float(b)
                elif b.ndim == 1:
                    if b.shape[0] == B:
                        b = b.reshape(B, 1).astype(np.float64, copy=False)
                    elif b.shape[0] == 1:
                        b = float(b[0])
                    else:
                        raise ValueError(f"self.b has shape {b.shape}; expected (B,), (1,), scalar, or (B,1).")
                elif b.ndim == 2:
                    if b.shape == (B, 1):
                        b = b.astype(np.float64, copy=False)
                    elif b.shape == (1, 1):
                        b = float(b[0, 0])
                    else:
                        raise ValueError(f"self.b has shape {b.shape}; expected (B,1), (1,1), or scalar.")
                else:
                    raise ValueError(f"self.b must be scalar or <=2D array; got ndim={b.ndim}.")
            else:
                # e.g., python float/int is handled by scalar branch above; everything else is unsupported
                raise TypeError(f"Unsupported type for self.b: {type(b)}")

        # ---- Sort values ----
        X_sort = np.sort(X64, axis=1)  # (B, N)

        # Index of last loss for each batch (<= b)
        # L = count(<=b) - 1, can be -1 if nothing is <= b
        L = (X_sort <= b).sum(axis=1) - 1  # (B,)

        # Index row [1, N] broadcasts to [B, N]
        idx = np.arange(N, dtype=np.int64)[None, :]  # (1, N)

        mask_minus = idx <= L[:, None]     # losses
        mask_plus = ~mask_minus            # gains

        # ---- Quantiles (float64) ----
        I = np.arange(1, N + 1, dtype=np.float64)  # 1..N
        z1 = (N + 1.0 - I) / N
        z2 = (N - I) / N
        z3 = I / N
        z4 = (I - 1.0) / N

        # ---- Utility and weighting ----
        # Assumes these accept NumPy arrays and return NumPy arrays.
        u_plus = self.u_plus(X_sort)  # (B, N)
        u_minus = self.u_neg(X_sort)  # (B, N)

        # Weight increments are (N,) and broadcast across batch to (B, N)
        w_plus_inc = (self.w_plus(z1) - self.w_plus(z2))  # (N,)
        w_neg_inc  = (self.w_neg(z3)  - self.w_neg(z4))   # (N,)

        rho_plus  = u_plus  * w_plus_inc[None, :] * mask_plus.astype(np.float64)
        rho_minus = u_minus * w_neg_inc[None, :]  * mask_minus.astype(np.float64)

        rho = rho_plus.sum(axis=1) - rho_minus.sum(axis=1)  # (B,)

        if getattr(self, "offset_ref", False):
            # Add reference point back in (vector or scalar)
            if np.isscalar(b):
                rho = rho + b
            else:
                rho = rho + np.squeeze(b, axis=1)  # (B,)

        if getattr(self, "mean_value_ref", False):
            self.b = None

        return rho.astype(orig_dtype, copy=False)

    def _pt_sample_expectation_batch(self, X, b = None):
        """Estimates CPT-value from only samples (no probs) in batch form"""
        assert isinstance(X, torch.Tensor), f"X must be a tensor, got {type(X)}"
        assert X.ndim == 2, f"X must be 2D array of value samples (B,N), got {X.shape}"
        assert not (self.mean_value_ref and b is not None), "Cannot provide b when mean_value_ref is True."
        B, N = X.shape
        device = X.device
        _dtype = X.dtype

        if self.mean_value_ref:
            self.b = torch.mean(X, dim=1, keepdim=True) # (B, 1)
        # elif b is not None:
        #     assert isinstance(b,torch.Tensor), "b must torch.Tensor if supplied at each timestep"
        #     self.b = b.reshape(B,1) # (B, 1)



        dtype = torch.float64
        X.to(dtype=torch.float64)

        # Ensure contiguous (helps some kernels, especially after slicing/gathering)
        X = X.contiguous()

        # ---- Sort values and align probabilities ----
        X_sort, sorted_idxs = torch.sort(X, dim=1)  # [B, N]

        # Index of last loss for each batch (<= b)
        L = (X_sort <= self.b).sum(dim=1) - 1  # [B]

        # Precompute (or reuse from self if N fixed) index row: [1, N] broadcasts to [B, N]
        idx = torch.arange(N, device=device).view(1, -1)

        # mask_minus: losses; mask_plus: gains
        mask_minus = idx <= L.unsqueeze(1)  # [B, N]
        mask_plus = ~mask_minus  # [B, N]

        # ---- Cumulative probabilities ----
        # # Forward cumsum for losses
        # # F_minus = P_sort.cumsum(dim=1)  # [B, N]
        #
        # # Reverse cumsum for gains
        # # P_rev = torch.flip(P_sort, dims=[1])
        # F_plus = torch.flip(P_rev.cumsum(dim=1), dims=[1])  # [B, N]
        #
        # # Select correct cumulative probs per entry without extra masks
        # F_plus = torch.clamp(F_plus, 0.0, 1.0)
        # F_minus = torch.clamp(F_minus, 0.0, 1.0)
        # Fk = torch.where(mask_minus, F_minus, F_plus)  # [B, N]

        # ---- Shifted Fk for decision weights ----
        # pad0 = X.new_zeros(B, 1)  # correct device & dtype

        # Keep the same structure as your original implementation
        # z1 = torch.cat([Fk, pad0], dim=1)[:, :-1]  # [B, N]
        # z2 = torch.cat([Fk, pad0], dim=1)[:, 1:]  # [B, N]
        # z3 = torch.cat([pad0, Fk], dim=1)[:, 1:]  # [B, N]
        # z4 = torch.cat([pad0, Fk], dim=1)[:, :-1]  # [B, N]

        # Compute quantiles
        I = torch.arange(1, N + 1, device=device, dtype=dtype)

        z1 = (N + 1 - I) / N
        z2 = (N - I) / N
        z3 = I / N
        z4 = (I - 1) / N

        # _b = torch.tensor(self.b, device=X.device, dtype=X.dtype)

        # ---- Utility and weighting ----
        u_plus = self.u_plus(X_sort)
        u_minus = self.u_neg(X_sort)

        rho_plus = u_plus * (self.w_plus(z1) - self.w_plus(z2)) * mask_plus
        rho_minus = u_minus * (self.w_neg(z3) - self.w_neg(z4)) * mask_minus

        # Only gains contribute to rho_plus, only losses to rho_minus
        # rho_plus = torch.nan_to_num(rho_plus, nan=0.0, posinf=0.0, neginf=0.0) * mask_plus
        # rho_minus = torch.nan_to_num(rho_minus, nan=0.0, posinf=0.0, neginf=0.0) * mask_minus

        # Final CPT expectation per batch element
        rho = rho_plus.sum(dim=1) - rho_minus.sum(dim=1)  # [B]

        if self.offset_ref:
            rho = rho + (self.b.squeeze()  if isinstance(self.b,torch.Tensor) else self.b)

        if self.mean_value_ref:
            self.b = None

        return rho.to(dtype=_dtype)

    def sample_expectation(self, X):
        raise NotImplementedError("Use batch version instead.")
        """Estimates CPT-value from only samples (no probs)"""
        if isinstance(X, torch.Tensor): is_torch = True
        elif isinstance(X, np.ndarray): is_torch = False
        elif isinstance(X, list): X = np.array(X); is_torch = False
        else:  raise ValueError("X must be a torch.Tensor, np.ndarray, or list.")
        assert X.ndim == 1, f"X must be 1D array of value samples, got {X.shape}"
        N_max = X.shape[0]

        if is_torch:
            device, dtype = X.device, X.dtype
            X_sort = torch.sort(X).values
            I = torch.arange(1, N_max + 1, device=device, dtype=dtype)
            k = (X_sort <= self.b).sum(dim=-1) - 1
            # if not torch.any(X_sort <= self.b):
            #     k = 0  # all gains
            # else:
            #     k = torch.where(X_sort <= self.b)[0][-1]
        else:
            X_sort = np.sort(X)
            I = np.arange(1, N_max + 1)
            if not np.any(X_sort <= self.b):
                k = 0  # all gains
            else:
                k = np.where(X_sort <= self.b)[0][-1]

        # Compute quantiles
        z_1 = (N_max + 1 - I) / N_max  # # {z_1 = (N_max + i - 1) / N_max <-- mistake}
        z_2 = (N_max - I) / N_max
        z_3 = I / N_max
        z_4 = (I - 1) / N_max

        rho_plus = self.u_plus(X_sort) * (self.w_plus(z_1) - self.w_plus(z_2))
        rho_minus = self.u_neg(X_sort) * (self.w_neg(z_3) - self.w_neg(z_4))
        # rho = torch.sum(rho_plus) - torch.sum(rho_minus)
        return rho_plus, rho_minus, k


    def expectation(self, values, p_values):
        """
        Applies the CPT-expectation multiple prospects (i.e. a series of value-probability pairs) which can arbitrarily
        replace the rational expectation operator E[v,p] = Σ(p*v). When dealing with more than two prospects, we must
        calculate the expectation over the cumulative probability distributions.
        :param values:
        :param p_values:
        :return:
        """
        raise NotImplementedError("Use batch version instead.")
        if self.is_rational:
            # Rational Expectation
            return np.sum(values * p_values)

        # Step 1: arrange all samples in ascending order and get indexs of gains/losses
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]

        K = len(sorted_v)  # number of samples
        if K == 1: return sorted_v[0]  # Single prospect = no CPT
        l = np.where(sorted_v <= self.b)[0]
        l = -1 if len(l) == 0 else l[-1]  # of no losses l=-1 indicator

        # Step 2: Calculate the cumulative liklihoods for gains and losses
        Fk = [min([max([0, np.sum(sorted_p[0:i + 1])]), 1]) for i in range(l + 1)] + \
             [min([max([0, np.sum(sorted_p[i:K])]), 1]) for i in range(l + 1, K)]  # cumulative probability
        Fk = Fk + [0]  # padding to make dealing with only gains or only losses easier

        # Step 3: Calculate biased expectation for gains and losses
        rho_p = self.perc_util_plus(sorted_v, Fk, l, K)
        rho_n = self.perc_util_neg(sorted_v, Fk, l, K)

        # Step 3: Add the cumulative expectation and return
        rho = rho_p - rho_n

        return rho


        # if add_pad:
        #     from itertools import zip_longest
        #     X = list(zip(*zip_longest(*X, fillvalue=0)))
        #     p_values = list(zip(*zip_longest(*p_values, fillvalue=0)))
        #
        #     X = torch.tensor(X, device=device, dtype=torch.float64)
        #     p_values = torch.tensor(p_values,device=device, dtype=torch.float64)

    def expectation_batch(self, X: torch.Tensor, p: torch.Tensor, add_pad = False,pad_val = torch.inf) -> torch.Tensor:
        """
        Calculates CPT expectation in batches from prospects of size (B, N)
        where B is batch size and N is number of prospects.
        """
        raise NotImplementedError("Use batch version instead.")
        if add_pad:
            from itertools import zip_longest
            device = X[0].device
            X = list(zip(*zip_longest(*X, fillvalue=pad_val)))
            p_values = list(zip(*zip_longest(*p, fillvalue=0)))
            X = torch.tensor(X, device=device, dtype=torch.float64)
            p = torch.tensor(p_values,device=device, dtype=torch.float64)

        assert isinstance(X, torch.Tensor)
        assert isinstance(p, torch.Tensor)

        B, N = X.shape
        device = X.device
        _dtype = X.dtype

        dtype = torch.float64
        X.to(dtype=torch.float64)
        p.to(dtype=torch.float64)

        if self.compiled_expectation:
            pt_params = [torch.tensor(param, dtype=torch.float32, device=device)
                         for param in [self.b, self.lam, self.eta_p, self.eta_n, self.delta_p, self.delta_n]]
            rho = _compiled_expectation(X, p, *pt_params)
            return rho

        # Ensure contiguous (helps some kernels, especially after slicing/gathering)
        X = X.contiguous()
        p = p.contiguous()

        # ---- Sort values and align probabilities ----
        X_sort, sorted_idxs = torch.sort(X, dim=1)  # [B, N]
        P_sort = torch.gather(p.to(device=device, dtype=dtype), 1, sorted_idxs)

        # Index of last loss for each batch (<= b)
        L = (X_sort <= self.b).sum(dim=1) - 1  # [B]

        # Precompute (or reuse from self if N fixed) index row: [1, N] broadcasts to [B, N]
        idx = torch.arange(N, device=device).view(1, -1)

        # mask_minus: losses; mask_plus: gains
        mask_minus = idx <= L.unsqueeze(1)  # [B, N]
        mask_plus = ~mask_minus  # [B, N]
        # Note: when all gains, L = -1 → mask_minus all False, mask_plus all True
        #       when all losses, L = N-1 → mask_minus all True, mask_plus all False

        # ---- Cumulative probabilities ----
        # Forward cumsum for losses
        F_minus = P_sort.cumsum(dim=1)  # [B, N]

        # Reverse cumsum for gains
        P_rev = torch.flip(P_sort, dims=[1])
        F_plus = torch.flip(P_rev.cumsum(dim=1), dims=[1])  # [B, N]

        # Select correct cumulative probs per entry without extra masks
        F_plus = torch.clamp(F_plus, 0.0, 1.0)
        F_minus = torch.clamp(F_minus, 0.0, 1.0)
        Fk = torch.where(mask_minus, F_minus, F_plus)  # [B, N]

        # ---- Shifted Fk for decision weights ----
        pad0 = X.new_zeros(B, 1)  # correct device & dtype

        # Keep the same structure as your original implementation
        z1 = torch.cat([Fk, pad0], dim=1)[:, :-1]  # [B, N]
        z2 = torch.cat([Fk, pad0], dim=1)[:, 1:]  # [B, N]
        z3 = torch.cat([pad0, Fk], dim=1)[:, 1:]  # [B, N]
        z4 = torch.cat([pad0, Fk], dim=1)[:, :-1]  # [B, N]
        _b = torch.tensor(self.b, device=X.device, dtype=X.dtype)

        # ---- Utility and weighting ----
        u_plus = self.u_plus(X_sort)
        u_minus = self.u_neg(X_sort)

        rho_plus    = u_plus * (self.w_plus(z1) - self.w_plus(z2))
        rho_minus   = u_minus * (self.w_neg(z3) - self.w_neg(z4))

        # Only gains contribute to rho_plus, only losses to rho_minus
        rho_plus =  torch.nan_to_num(rho_plus,  nan=0.0, posinf=0.0,neginf=0.0) * mask_plus
        rho_minus = torch.nan_to_num(rho_minus, nan=0.0, posinf=0.0,neginf=0.0) * mask_minus

        # Final CPT expectation per batch element
        rho = rho_plus.sum(dim=1) - rho_minus.sum(dim=1)  # [B]

        # Handle single-prospect case (no bias)

        # rho[single_prospect_mask] = X[single_prospect_mask,0]

        return rho.to(dtype=_dtype)


    def perc_util_plus(self, sorted_v, Fk, l, K):
        """Calculates the cumulative expectation of all utilities percieved as gains"""
        rho_p = 0
        for i in range(l + 1, K):
            rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # for i in range(l + 1, K - 1):
        #     rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # rho_p += self.u_plus(sorted_v[K - 1]) * self.w_plus(sorted_p[K - 1])
        return rho_p

    def perc_util_neg(self, sorted_v, Fk, l, K):
        """Calculates the cumulative expectation of all utilities percieved as losses"""
        # Fk =  Fk + [0]  # add buffer which results in commented out version below
        rho_n = 0
        for i in range(0, l + 1):
            rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        return rho_n
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # rho_n = self.u_neg(sorted_v[0]) * self.w_neg(sorted_p[0])
        # for i in range(1, l + 1):
        #     rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        # return rho_n

    def u_plus(self, v):
        """ Weights the values (v) perceived as losses (v>b)"""
        if isinstance(v,torch.Tensor):
            return torch.pow(torch.abs(v - self.b), self.eta_p)
        return np.power(np.abs(v - self.b), self.eta_p)

    def u_neg(self, v):
        """ Weights the values (v) perceived as gains (v<=b)"""
        if isinstance(v,torch.Tensor):
            return self.lam * torch.pow(torch.abs(v - self.b), self.eta_n)
        return self.lam * np.power(np.abs(v - self.b), self.eta_n)

    def w_plus(self, p):
        """ Weights the probabilities p for probabilities of values perceived as gains  (v>b)"""
        return self._w(p, self.delta_p)

    def w_neg(self, p):
        """ Weights the probabilities p for probabilities of values perceived as losses (v<=b)"""
        return self._w(p, self.delta_n)

    def _w(self, p, delta):
        if isinstance(p,torch.Tensor):
            z = torch.pow(p, delta)  # precompute term
            denom = z + torch.pow(1 - p, delta)
            denom = torch.pow(denom, 1 / delta)
            return z / denom

        z = np.power(p,delta)  # precompute term
        denom = z + np.power(1 - p,delta)
        denom = np.power(denom, 1 / delta)
        return z / denom
        # return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))

    def __repr__(self):
        return f"CPT(b:{self.b}, λ:{self.lam}, η+:{self.eta_p}, η-:{self.eta_n}, δ+:{self.delta_p}, δ-:{self.delta_n})"


class CumulativeProspectTheory_OLD:
    def __init__(self, b, lam, eta_p, eta_n, delta_p, delta_n):
        """
        Instantiates a CPT object that can be used to model human risk-sensitivity.
        :param b: reference point determining if outcome is gain or loss
        :param lam: loss-aversion parameter
        :param eta_p: exponential gain on positive outcomes
        :param eta_n: exponential loss on negative outcomes
        :param delta_p: probability weighting for positive outcomes
        :param delta_n: probability weighting for negative outcomes
        """
        # assert b==0, "Reference point must be 0"
        self.b = b
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

        self.expected_td_targets = np.zeros(256, dtype=np.float32)
        self._f_vmap = None

        self.is_rational = True
        if self.b != 0:
            self.is_rational = False
        elif self.lam != 1:
            self.is_rational = False
        elif self.eta_p != 1:
            self.is_rational = False
        elif self.eta_n != 1:
            self.is_rational = False
        elif self.delta_p != 1:
            self.is_rational = False
        elif self.delta_n != 1:
            self.is_rational = False

    def sample_expectation_batch(self, X):
        """Estimates CPT-value from only samples (no probs) in batch form"""
        if isinstance(X, torch.Tensor): is_torch = True
        elif isinstance(X, np.ndarray): is_torch = False
        elif isinstance(X, list): X = np.array(X); is_torch = False
        else:  raise ValueError("X must be a torch.Tensor, np.ndarray, or list.")
        assert X.ndim == 2, f"X must be 2D array of value samples (B,N), got {X.shape}"

        N_max = X.shape[1]
        B = X.shape[0]

        if is_torch:
            device, dtype = X.device, X.dtype
            X = X.to(device=device, dtype=torch.float64)  # change to double percision
            X_sort = torch.sort(X, dim=1).values
            I = torch.arange(1, N_max + 1, device=device, dtype=dtype).unsqueeze(0).repeat(B,1)
            K = (X_sort <= self.b).sum(dim=1)  -1

            # Simpler version for rational
            # if self.is_rational:
            #     return quantile_expectation_batch_torch(X)
        else:
            raise NotImplementedError("Numpy batch version not implemented yet.")
            # X_sort = np.sort(X, axis=1)
            # I = np.arange(1, N_max + 1).reshape(1,-1).repeat(B,axis=0)
            # k = np.where(X_sort <= self.b, 1, 0).sum(axis=1) - 1
            # k = np.clip(k,min=0)

        # Compute quantiles
        z_1 = (N_max + 1 - I) / N_max
        z_2 = (N_max - I) / N_max
        z_3 = I / N_max
        z_4 = (I - 1) / N_max

        rho_plus = self.u_plus(X_sort) * (self.w_plus(z_1) - self.w_plus(z_2))
        rho_minus = self.u_neg(X_sort) * (self.w_neg(z_3) - self.w_neg(z_4))

        if is_torch:
            # Create index mask [B, N]
            idx = torch.arange(N_max, device=K.device).unsqueeze(0).expand(B, N_max)

            # Compute masks for each term
            mask_plus = idx > K.unsqueeze(1)  # True for j >= k[b]
            mask_minus = idx <= K.unsqueeze(1)  # True for j < k[b]
            mask_all_gains = K.unsqueeze(1) == -1  # True for all gains

            mask_plus = mask_plus | mask_all_gains
            mask_minus = mask_minus & (~mask_all_gains)

            # Apply masks and sum along dim=1
            rho = (rho_plus * mask_plus).sum(dim=1) - (rho_minus * mask_minus).sum(dim=1)

            # rho = torch.zeros(B, device=device, dtype=dtype)
            # for b in range(B):
            #     k = K[b]
            #     if k == -1:  rho[b] = torch.sum(rho_plus[b, :]) # all gains
            #     else: rho[b] = torch.sum(rho_plus[b, k:]) - torch.sum(rho_minus[b, :k])
            rho = rho.to(dtype=dtype)
        else:
            rho = np.zeros(B)
            for b in range(B):
                rho[b] = np.sum(rho_plus[b, K[b]:]) - np.sum(rho_minus[b, :K[b]])

        # assert rho.shape == (B,), f"rho must be of shape (B,), got {rho.shape}"
        return rho

    def sample_expectation_vmap(self, X):
        assert X.ndim == 2, f"X must be 2D array of value samples (B,N), got {X.shape}"
        assert isinstance(X,torch.Tensor), f"X must be a tensor, got {type(X)}"
        B,N = X.shape
        if self._f_vmap is None:
            self._f_vmap = torch.vmap(self._vec_sample_expectation)
        # return self._f_vmap(X)
        rho_plus, rho_minus, K = self._f_vmap(X)

        # Create index mask [B, N]
        # Create index mask [B, N]
        idx = torch.arange(N, device=K.device).unsqueeze(0).expand(B, N)

        # Compute masks for each term
        mask_plus = idx > K.unsqueeze(1)  # True for j >= k[b]
        mask_minus = idx <= K.unsqueeze(1)  # True for j < k[b]
        mask_all_gains = K.unsqueeze(1) == -1  # True for all gains

        mask_plus = mask_plus | mask_all_gains
        mask_minus = mask_minus & (~mask_all_gains)

        # Apply masks and sum along dim=1
        rho = (rho_plus * mask_plus).sum(dim=1) - (rho_minus * mask_minus).sum(dim=1)
        return rho


    # @torch.vmap
    def _vec_sample_expectation(self, X):
        """helper function for vectorized sample expectation (TORCH ONLY)"""
        assert X.ndim == 1, f"X must be 1D array of value samples, got {X.shape}"
        N_max = X.shape[0]
        device, dtype = X.device, X.dtype
        X_sort = torch.sort(X).values
        I = torch.arange(1, N_max + 1, device=device, dtype=dtype)
        K = (X_sort <= self.b).sum(dim=-1) - 1

        # Compute quantiles
        z_1 = (N_max + 1 - I) / N_max  # # {z_1 = (N_max + i - 1) / N_max <-- mistake}
        z_2 = (N_max - I) / N_max
        z_3 = I / N_max
        z_4 = (I - 1) / N_max

        rho_plus = self.u_plus(X_sort) * (self.w_plus(z_1) - self.w_plus(z_2))
        rho_minus = self.u_neg(X_sort) * (self.w_neg(z_3) - self.w_neg(z_4))

        return rho_plus, rho_minus, K
        idx = torch.arange(N_max, device=K.device).unsqueeze(0)

        # Compute masks for each term
        mask_plus = idx > K.unsqueeze(1)  # True for j >= k[b]
        mask_minus = idx <= K.unsqueeze(1)  # True for j < k[b]
        mask_all_gains = K.unsqueeze(1) == -1  # True for all gains

        mask_plus = mask_plus | mask_all_gains
        mask_minus = mask_minus & (~mask_all_gains)

        # Apply masks and sum along dim=1
        rho = (rho_plus * mask_plus).sum(dim=1) - (rho_minus * mask_minus).sum(dim=1)
        # Create index mask [B, N]
        # idx = torch.arange(N_max, device=K.device).unsqueeze(0).expand(B, N_max)
        #
        # # Compute masks for each term
        # mask_plus = idx > K.unsqueeze(1)  # True for j >= k[b]
        # mask_minus = idx <= K.unsqueeze(1)  # True for j < k[b]
        # mask_all_gains = K.unsqueeze(1) == -1  # True for all gains
        #
        # mask_plus = mask_plus | mask_all_gains
        # mask_minus = mask_minus & (~mask_all_gains)
        #
        # # Apply masks and sum along dim=1
        # rho = (rho_plus * mask_plus).sum(dim=1) - (rho_minus * mask_minus).sum(dim=1)
        return rho

    def sample_expectation(self, X):
        """Estimates CPT-value from only samples (no probs)"""
        if isinstance(X, torch.Tensor): is_torch = True
        elif isinstance(X, np.ndarray): is_torch = False
        elif isinstance(X, list): X = np.array(X); is_torch = False
        else:  raise ValueError("X must be a torch.Tensor, np.ndarray, or list.")
        assert X.ndim == 1, f"X must be 1D array of value samples, got {X.shape}"
        N_max = X.shape[0]

        if is_torch:
            device, dtype = X.device, X.dtype
            X_sort = torch.sort(X).values
            I = torch.arange(1, N_max + 1, device=device, dtype=dtype)
            k = (X_sort <= self.b).sum(dim=-1) - 1
            # if not torch.any(X_sort <= self.b):
            #     k = 0  # all gains
            # else:
            #     k = torch.where(X_sort <= self.b)[0][-1]
        else:
            X_sort = np.sort(X)
            I = np.arange(1, N_max + 1)
            if not np.any(X_sort <= self.b):
                k = 0  # all gains
            else:
                k = np.where(X_sort <= self.b)[0][-1]

        # Compute quantiles
        z_1 = (N_max + 1 - I) / N_max  # # {z_1 = (N_max + i - 1) / N_max <-- mistake}
        z_2 = (N_max - I) / N_max
        z_3 = I / N_max
        z_4 = (I - 1) / N_max

        rho_plus = self.u_plus(X_sort) * (self.w_plus(z_1) - self.w_plus(z_2))
        rho_minus = self.u_neg(X_sort) * (self.w_neg(z_3) - self.w_neg(z_4))
        # rho = torch.sum(rho_plus) - torch.sum(rho_minus)
        return rho_plus, rho_minus, k

        # rho_plus = self.u_plus(X_sort[k:]) * (self.w_plus(z_1[k:]) - self.w_plus(z_2[k:]))
        # rho_minus = self.u_neg(X_sort[:k]) * (self.w_neg(z_3[:k]) - self.w_neg(z_4[:k]))
        #
        # if is_torch:
        #     rho = torch.sum(rho_plus) - torch.sum(rho_minus)
        # else:
        #     rho = np.sum(rho_plus) - np.sum(rho_minus)
        # return rho

    def expectation(self, values, p_values):
        """
        Applies the CPT-expectation multiple prospects (i.e. a series of value-probability pairs) which can arbitrarily
        replace the rational expectation operator E[v,p] = Σ(p*v). When dealing with more than two prospects, we must
        calculate the expectation over the cumulative probability distributions.
        :param values:
        :param p_values:
        :return:
        """
        if self.is_rational:
            # Rational Expectation
            return np.sum(values * p_values)
        if self.mean_value_ref:
            self.b = np.mean(values)

        # Step 1: arrange all samples in ascending order and get indexs of gains/losses
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]

        K = len(sorted_v)  # number of samples
        if K == 1: return sorted_v[0]  # Single prospect = no CPT
        l = np.where(sorted_v <= self.b)[0]
        l = -1 if len(l) == 0 else l[-1]  # of no losses l=-1 indicator

        # Step 2: Calculate the cumulative liklihoods for gains and losses
        Fk = [min([max([0, np.sum(sorted_p[0:i + 1])]), 1]) for i in range(l + 1)] + \
             [min([max([0, np.sum(sorted_p[i:K])]), 1]) for i in range(l + 1, K)]  # cumulative probability
        Fk = Fk + [0]  # padding to make dealing with only gains or only losses easier

        # Step 3: Calculate biased expectation for gains and losses
        rho_p = self.perc_util_plus(sorted_v, Fk, l, K)
        rho_n = self.perc_util_neg(sorted_v, Fk, l, K)

        # Step 3: Add the cumulative expectation and return
        rho = rho_p - rho_n

        return rho

    def perc_util_plus(self, sorted_v, Fk, l, K):
        """Calculates the cumulative expectation of all utilities percieved as gains"""
        rho_p = 0
        for i in range(l + 1, K):
            rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # for i in range(l + 1, K - 1):
        #     rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # rho_p += self.u_plus(sorted_v[K - 1]) * self.w_plus(sorted_p[K - 1])
        return rho_p

    def perc_util_neg(self, sorted_v, Fk, l, K):
        """Calculates the cumulative expectation of all utilities percieved as losses"""
        # Fk =  Fk + [0]  # add buffer which results in commented out version below
        rho_n = 0
        for i in range(0, l + 1):
            rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        return rho_n
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # rho_n = self.u_neg(sorted_v[0]) * self.w_neg(sorted_p[0])
        # for i in range(1, l + 1):
        #     rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        # return rho_n

    def u_plus(self, v):
        """ Weights the values (v) perceived as losses (v>b)"""
        if isinstance(v,torch.Tensor):
            return torch.pow(torch.abs(v - self.b), self.eta_p)
        return np.power(np.abs(v - self.b), self.eta_p)

    def u_neg(self, v):
        """ Weights the values (v) perceived as gains (v<=b)"""
        if isinstance(v,torch.Tensor):
            return self.lam * torch.pow(torch.abs(v - self.b), self.eta_n)
        return self.lam * np.power(np.abs(v - self.b), self.eta_n)

    def w_plus(self, p):
        """ Weights the probabilities p for probabilities of values perceived as gains  (v>b)"""
        return self._w(p, self.delta_p)

    def w_neg(self, p):
        """ Weights the probabilities p for probabilities of values perceived as losses (v<=b)"""
        return self._w(p, self.delta_n)

    def _w(self, p, delta):
        if isinstance(p,torch.Tensor):
            z = torch.pow(p, delta)  # precompute term
            denom = z + torch.pow(1 - p, delta)
            denom = torch.pow(denom, 1 / delta)
            return z / denom

        z = np.power(p,delta)  # precompute term
        denom = z + np.power(1 - p,delta)
        denom = np.power(denom, 1 / delta)
        return z / denom
        # return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))

def quantile_expectation2(X,keepdim=False):
    """
    Quantile-based expectation from samples.

    Computes  ∫_0^1 Q_X(u) d w(u)  using the empirical quantile function Q_X
    (order statistics) and Riemann–Stieltjes sum:
        E_w[X] ≈ Σ_{i=1}^n x_(i) * ( w(i/n) - w((i-1)/n) ),
    where x_(i) are the sorted samples along `axis`.
    """
    # raise NotImplementedError("Use pdf_expectation instead; quantile_expectation is deprecated.")
    # assert X.ndim == 1, "X must be 1D array of value samples"
    #
    X_sorted = np.sort(X, axis=-1)
    n = X.shape[0]
    Xi = X[:-1]
    exp = 0
    for i in range(n):
        # exp += X_sorted[i] * ((n+1-i) / n - (n-i) / n)
        exp += X_sorted[i] * ((n + 1 - i) / n - (n - i) / n)
    return exp
def quantile_expectation_batch_torch(X,keepdim=False):
    """
    Batched quantile-based expectation from samples.

    Computes  ∫_0^1 Q_X(u) d w(u)  using the empirical quantile function Q_X
    (order statistics) and Riemann–Stieltjes sum:
        E_w[X] ≈ Σ_{i=1}^n x_(i) * ( w(i/n) - w((i-1)/n) ),
    where x_(i) are the sorted samples along `axis`.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D tensor (B, N).")

    B, N = X.shape
    device, dtype = X.device, X.dtype
    X = X.to(device=device, dtype=torch.float64)        # change to double percision

    # Sort each batch individually along dim=1
    X_sort, _ = torch.sort(X, dim=-1)
    # I = torch.arange(1, N + 1, device=device, dtype=dtype).unsqueeze(0).repeat(B,1)
    exp_vals = torch.zeros(B, device=device, dtype=dtype)
    for i in range(N):
        exp_vals += X_sort[:, i] * ((N + 1 - i) / N - (N - i) / N)

    return exp_vals.to(dtype=dtype)

# @njit
def pdf_expectation(X,pdf, normalize=True):
    """Computes expected mean of distribution from samples and pdf weights using trapezoidal integration."""
    # isort =  np.argsort(X)
    # X_sort = X[isort]
    # pdf_sort = pdf[isort]
    # dx = X_sort[1:] - X_sort[:-1]
    #
    # # Normalize pdf so that ∫ pdf dx = 1 over the uneven grid
    # if normalize:
    #     pdf_sort /= np.trapz(pdf_sort, X_sort)
    #
    # # Compute expected value E[X] = ∫ x * pdf(x) dx
    # exp = np.sum(0.5 * (X_sort[1:] + X_sort[:-1]) * pdf_sort[:-1] * dx)
    # return exp
    isort = np.argsort(X)
    X_sort = X[isort]
    pdf_sort = pdf[isort]
    dx = X_sort[1:] - X_sort[:-1]

    # Normalize pdf so that ∫ pdf dx = 1 over the uneven grid
    if normalize:
        pdf_sort /= np.trapz(pdf_sort, X_sort)

    # Compute expected value E[X] = ∫ x * pdf(x) dx
    # exp = np.sum(0.5 * (X_sort[1:] + X_sort[:-1]) * pdf_sort[:-1] * dx)
    exp = np.sum(0.5 * (X_sort[1:] + X_sort[:-1]) * pdf_sort[:-1] * dx)
    return exp

def pdf_expectation_torch_batch(X, pdf, normalize=True):
    """
    Batched expected value of a PDF over unevenly spaced X.

    Args:
        X   : (B, N) tensor of sample locations
        pdf : (B, N) tensor of pdf values
        normalize (bool): normalize pdf so ∫pdf dx = 1 for each batch

    Returns:
        exp_vals : (B,) tensor of expected values
    """
    if X.ndim != 2 or pdf.ndim != 2:
        raise ValueError("X and pdf must both be 2D tensors (B, N).")
    if X.shape != pdf.shape:
        raise ValueError("X and pdf must have the same shape.")

    B, N = X.shape
    device, dtype = X.device, X.dtype

    X = X.to(device=device, dtype=torch.float64)        # change to double percision
    pdf = pdf.to(device=device, dtype=torch.float64)    # change to double percision


    # Sort each batch individually along dim=1
    X_sort, sort_idx = torch.sort(X, dim=1)
    pdf_sort = torch.gather(pdf, 1, sort_idx)

    # compute area intervals and normalize pdfs
    dx = X_sort[:, 1:] - X_sort[:, :-1]
    if normalize:
        # Trapezoidal integral for normalization (batchwise)
        area = torch.sum(0.5 * (pdf_sort[:, 1:] + pdf_sort[:, :-1]) * dx, dim=1, keepdim=True)
        pdf_sort = pdf_sort / area

    # Midpoint-style expectation across uneven grid
    exp_vals = torch.sum(
        0.5 * (X_sort[:, 1:] + X_sort[:, :-1]) * pdf_sort[:, :-1] * dx,
        dim=1
    )

    # handle edge case where all samples are same reward (numerical instability)
    uniform_dx_idxs = torch.where(torch.all(dx==0,dim=1))
    for i in uniform_dx_idxs:
        exp_vals[i] = X_sort[i,0]
    assert torch.all(torch.isfinite(exp_vals)), "Non-finite expected values computed."

    return exp_vals.to(dtype=dtype)


def test_speed():
    print(f'\n\n########################################################')
    print(     "Speed Test Results: ####################################")
    import time
    N_TESTS = 100

    N_batch = 256
    n_samples = 100

    mu, std = 2.0, 0.1
    X = np.random.normal(mu, std, size=n_samples)
    cpt = CumulativeProspectTheory(b=0.0, lam=1, eta_p=1, eta_n=1, delta_p=1, delta_n=1)
    X = torch.tensor(X.reshape(1, -1), dtype=torch.float64,device='cuda')
    X = X.repeat(N_batch,1)



    ###################################################
    # f = torch.vmap(cpt.sample_expectation)
    start = time.time()
    for _ in range(N_TESTS):
        val = cpt.sample_expectation_vmap(X)
    end = time.time()
    print(f"CPT vmap Batch Time: {(end - start)/N_TESTS*1000:.3f} ms per {N_batch} samples [{val[0].item():.5f}]")

    ###################################################
    start = time.time()
    for _ in range(N_TESTS):
        quantile_expectation_batch_torch(X)
    end = time.time()
    print(f"Quantile Batch Time: {(end - start)/N_TESTS*1000:.3f} ms per {N_batch} samples [{val[0].item():.5f}]")

    ###################################################
    start = time.time()
    for _ in range(N_TESTS):
        cpt.sample_expectation_batch(X)
    end = time.time()
    print(f"CPT Batch Time: {(end - start)/N_TESTS*1000:.3f} ms per {N_batch} samples [{val[0].item():.5f}]")

def test_risk_sensitivity():
    print(f'\n\n########################################################')
    print(     "Accuracy Test Results: #################################")
    np.random.seed(0)
    # set scipy seed
    n_samples = 10000

    # cpt = CumulativeProspectTheory_Compiled(b=0.0, lam=2.25, eta_p=0.88, eta_n=0.88, delta_p=0.61, delta_n=0.69)
    cpt = CumulativeProspectTheory(b='mean', lam=1, eta_p=1, eta_n=1, delta_p=1, delta_n=1)

    # for mu, std in zip([2.0], [0.1]):

    for mu,std in zip([-2.0, 0.0, 0.5,2.0],[1.0,0.5,0.75,0.1]):
    # for mu, std in zip([5.0, 1,0.0,  0.5,0.75, -2.0], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]):
    # for mu, std in zip([5.0, 1,  0.5,0.0, 0.5, -1.0, -5], [1e-6,1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]):
    # for mu, std in zip([5.0, 1, 0.5, 0.0, 0.5, -1.0, -5],
    #                    [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]):
        X = np.random.normal(mu, std, size=n_samples)
        pdf = norm.pdf(X, loc=mu, scale=std)
        # pdf = np.clip(pdf,0,1)


        # print(f"Analytical E[X]: {exp_analytical}")
        print(f'\n mu = {mu} std ={std}----------------------------------')
        # quant_exp_err = mu - quantile_expectation(X) #mu - quantile_expectation(X)
        # quant_exp_err = mu - quantile_expectation2(X)  # mu - quantile_expectation(X)
        # pdf_exp_err = mu - pdf_expectation(X, pdf, normalize=True)
        # cpt_exp_err = mu - cpt.expectation(X, pdf/np.sum(pdf))




        X = torch.tensor(X.reshape(1,-1), dtype=torch.float64)
        pdf = torch.tensor(pdf.reshape(1,-1), dtype=torch.float64)
        X = X.repeat(256, 1)
        pdf = pdf.repeat(256, 1)

        cpt_np_batch = cpt.sample_expectation_batch(X.numpy())
        cpt_torch_batch = cpt.sample_expectation_batch(X).numpy()

        assert np.all(np.isclose(cpt_np_batch, cpt_torch_batch)), f"CPT batch and numpy batch results do not match! {cpt_np_batch[0]-cpt_torch_batch[0]}"
        # ----------------
        # pdf_pt_exp_err = (mu - pdf_expectation_torch_batch(X, pdf, normalize=True)).numpy()[0]

        # cpt_exp_err_batch = mu - cpt.sample_expectation_batch(X).numpy()[0]
        # cpt_exp_err_vmap = mu - cpt.sample_expectation_vmap(X).numpy()[0]
        # quant_exp_err = mu - quantile_expectation_batch_torch(X).numpy()[0]
        #
        # print(f"CPT batch Error E[X]: {cpt_exp_err_batch}")
        # print(f"CPT vmap Error E[X]: {cpt_exp_err_vmap}")
        # print(f"Quantile Int Error E[X]: {quant_exp_err}")
        # # print(f"PDF-weighted Error E[X]: {pdf_exp_err}")
        # # print(f"PT-PDF-weighted Error E[X]: {pdf_pt_exp_err}")
        # best =['CPT','Quantiale','PDF'][ np.argmax([cpt_exp_err_batch,quant_exp_err,pdf_exp_err])]
        # print(f"Best Method: {best}")
        # # print(f"PDF-weighted Error E[X]: {mu - pdf_expectation(X, pdf, normalize=False )}")
        # # print(f"PDF-weighted Error E[X]: {mu - pdf_expectation2(X, pdf)}")


if __name__ == '__main__':
    test_risk_sensitivity()
    test_speed()
