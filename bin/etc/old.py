class NormLayer(Layer):
    pNorm      = 2
    removeBias = True
    takeExp    = False
    dim        = -1
    lr         = 0.001

    def __init__(self, nInput, nOutput, pNorm=2, exp=False):
        """
        Hand-coded implementation of Layer Normalization or Softmax-like operation.
        Operates on the last (feature) dimension, supporting both 2D and 3D input.
        """
        super().__init__(nInput,nOutput)
        self.pNorm  = pNorm
        self.exp    = exp
        self.dim    = -1
        # predict the bias and variance within this layer
        self.W      = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        shape = x.shape
        N = shape[self.dim]
        mean = x.mean(dim=self.dim, keepdim=True)
        if self.pNorm == 2:
            assert(self.exp == False)
            norm     = torch.sum((x - mean) ** 2, dim=self.dim, keepdim=True) / (N - 1)
            norm     = torch.sqrt(norm + epsilon)
            moments  = torch.concat([mean, norm], axis=self.dim)
            moments -= self.W
            x_norm   = (x - moments[..., 0:1]) / moments[..., 1:2]
        else:
            assert(self.pNorm == 1)
            if self.exp:
                exp_x   = torch.exp(x)
                sum_exp = exp_x.sum(dim=self.dim, keepdim=True)
                x_norm  = exp_x / (sum_exp + epsilon)
                moments = sum_exp
            else:
                sum_x  = torch.sum(x, dim=self.dim, keepdim=True)
                x_norm = x / (sum_x + epsilon)
                moments = sum_x
            raise NotImplementedError(
                "NormLayer pNorm=1 requires W prediction support in this branch."
            )
        # append mean and norm
        output = torch.concat((x_norm, moments), dim=-1)
        return output

    def reverse(self, y):
        """
        Reverse operation (only defined for pNorm=2 without softmax).
        """
        if self.pNorm != 2:
            raise NotImplementedError("Reverse only supported for pNorm=2.")
        x    = y[...,  0:-2]
        mean = y[..., -2:-1] + self.W[0:1]
        norm = y[..., -1:]   + self.W[1:2]
        return x * norm + mean

    @staticmethod
    def test():
        torch.manual_seed(42)

        # === Test 1: pNorm=2, 2D forward + reverse ===
        x = torch.randn(10, 20, device=TheDevice.get())
        layer = NormLayer(20, 22, pNorm=2)
        layer.lr = 0
        normalized = layer(x)
        assert normalized.shape == (10, 22), f"2D shape: expected (10,22), got {normalized.shape}"
        reconstructed = layer.reverse(normalized)
        assert torch.allclose(x, reconstructed, atol=1e-5), "2D reverse failed"

        # === Test 2: pNorm=2, 3D forward + reverse ===
        x_3d = torch.randn(4, 5, 20, device=TheDevice.get())
        normalized_3d = layer(x_3d)
        assert normalized_3d.shape == (4, 5, 22), f"3D shape: expected (4,5,22), got {normalized_3d.shape}"
        reconstructed_3d = layer.reverse(normalized_3d)
        assert torch.allclose(x_3d, reconstructed_3d, atol=1e-5), "3D reverse failed"


class NewPiLayer(Layer):
    r"""Log-space multiplicative layer.

    Forward:
        y = exp(clamp(W @ log(clamp(x, ε, 1)) + b, log(ε), 0))

    Inputs are expected in (0, 1].  In log-space the layer is a simple
    affine map, and clamps keep the result in (0, 1].  Unlike OldPiLayer
    this does **not** require nOutput = 2 * nInput for invertibility —
    the invertible variant uses the same InvertibleLinearLayer but
    operates directly in log-space without interleaving.

    When ``stable=True`` (default) the clamp operations are applied.
    When ``stable=False`` the clamps are omitted (caller guarantees
    inputs are in range and the affine result stays in [log(ε), 0]).

    When ``invertible=True``:
        Reverse: x = exp(W_inv @ (log(clamp(y, ε, 1)) − b))
    """
    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.stable     = stable
        self.hasBias    = hasBias
        if invertible:
            self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                               naive=naive, ergodic=ergodic,
                                               stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=hasBias,
                                     naive=naive, ergodic=ergodic, stable=stable)
        self.layers.append(self.layer)
        # Log-space affine needs near-identity init so that the sum
        # W @ log(x) stays bounded.  Replace the default randn W with
        # eye + small noise.
        if not ergodic and hasattr(self.layer, 'W'):
            with torch.no_grad():
                self.layer.W.copy_(
                    torch.eye(nInput, nOutput) +
                    0.01 * torch.randn(nInput, nOutput)
                )

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def resample_noise(self):
        self.layer.resample_noise()

    def _effective_bias(self):
        """Bias to add to W @ log(x), or 0 if hasBias=False."""
        if not self.hasBias:
            return 0
        if self.layer.ergodic:
            return self.layer.bias * self.layer.biasWeight + self.layer.var * self.layer.biasNoise
        return self.layer.biasWeight

    def forward(self, x):
        if self.layer.ergodic:
            self.resample_noise()
        W = self.layer.compute_W_current()               # [nIn, nOut]
        x = x.to(W.device)
        # --- log-space affine: y = exp(clamp(W @ log(clamp(x)) + b)) ---
        if self.stable:
            x = x.clamp(min=0) + epsilon #x.clamp(min=epsilon, max=1.0)
        log_x = torch.log(x)                             # [..., nIn]
        wx = log_x @ W                                   # [..., nOut]
        b  = self._effective_bias()
        wx = wx + b
        if self.stable:
            wx = wx.clamp(min=math.log(epsilon), max=0.0)
        return torch.exp(wx)

    def reverse(self, y):
        """Recover x from y.  Requires invertible=True.

        x = exp(W_inv @ (log(clamp(y, ε, 1)) − b))
        """
        W_inv = self.layer.compute_Winverse_current()     # [nOut, nIn]
        y = y.to(W_inv.device)
        if self.stable:
            y = y.clamp(min=epsilon)
        log_y = torch.log(y)
        b = self._effective_bias()
        gamma = log_y - b                                 # [..., nOut]
        log_x = gamma @ W_inv                             # [..., nIn]
        if self.layer.ergodic:
            self.resample_noise()
        return torch.exp(log_x)

    @staticmethod
    def test():
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayer(nInput=nInput, nOutput=nOutput)
        device = next(layer.parameters()).device
        # Inputs in (0, 1]
        x = torch.rand((nBatch, 6, nInput), device=device).clamp(min=epsilon)
        layer.set_sigma(0.999)
        y = layer(x)
        assert y.shape == (nBatch, 6, nOutput), f"shape mismatch: {y.shape}"
        print(f"PiLayer forward: input {x.shape} -> output {y.shape}")

        def check_roundtrip(desc, **kwargs):
            kw = dict(nInput=3, nOutput=4, invertible=True)
            kw.update(kwargs)
            layer = PiLayer(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            inputs = [('3D [B,S,nIn]', torch.rand(16, 5, nI, device=device).clamp(min=epsilon)),
                      ('2D [B,nIn]',   torch.rand(16, nI,    device=device).clamp(min=epsilon))]
            for tag, x in inputs:
                layer.set_sigma(0.0)
                y = layer.forward(x)
                x_recon = layer.reverse(y)
                error = torch.norm(x - x_recon) / torch.norm(x)
                assert error < 1e-4, f"{desc} {tag}: reconstruction error {error:.2e}"
            print(f"  {desc}: OK")

        print("Invertible PiLayer (log-space) roundtrip variations:")
        check_roundtrip("naive=T hasBias=T", naive=True,  hasBias=True)
        check_roundtrip("naive=T hasBias=F", naive=True,  hasBias=False)
        check_roundtrip("naive=F hasBias=T", naive=False, hasBias=True)
        check_roundtrip("naive=F hasBias=F", naive=False, hasBias=False)
        check_roundtrip("square nIn=nOut=4", naive=True,  hasBias=False, nInput=4, nOutput=4)
        check_roundtrip("nOut<nIn 5->3",     naive=True,  hasBias=False, nInput=5, nOutput=3)
        check_roundtrip("ergodic naive=T hasBias=F", naive=True,  hasBias=False, ergodic=True)
        check_roundtrip("ergodic naive=T hasBias=T", naive=True,  hasBias=True,  ergodic=True)
        check_roundtrip("ergodic naive=F hasBias=T", naive=False, hasBias=True,  ergodic=True)
        print("PiLayer tests passed.")


class PiLayerOld(Layer):
    """Multiplicative layer: y_j = prod_i (1 + tanh(w_ji * x_i)).

    Forward materializes W via the inner layer, computes the outer product
    x.unsqueeze(-1) * W.unsqueeze(0) to keep per-input factors separate,
    applies tanh element-wise, then products via exp(sum(log(...))).
    No bias is applied inside the outer product.

    When ``invertible=False`` (default):
        y_j = exp(sum_i log(1 + tanh(w_ji * x_i)))

    When ``invertible=True``: outputs interleaved (y, z) pairs where
        y_j = exp(sum_i log(1 + tanh(w_ji * x_i)))
        z_j = exp(sum_i log(1 - tanh(w_ji * x_i)))
    Reverse: gamma_j = 0.5*(log y_j - log z_j) = sum_i x_i*w_ji = (x@W)_j,
    then x = gamma @ W_inv using the materialized inverse.

    All ergodic machinery lives in the inner layer; PiLayerOld dispatches
    the ergodic interface there via self.layers.
    """
    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.saturate   = True
        self.stable     = stable
        self.hasBias    = hasBias
        if invertible:
            self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                               naive=naive, ergodic=ergodic, stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=hasBias,
                                     naive=naive, ergodic=ergodic, stable=stable)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def resample_noise(self):
        self.layer.resample_noise()

    def _effective_bias(self):
        """Bias to add to WX, or None if hasBias=False."""
        if not self.hasBias:
            return 0
        if self.layer.ergodic:
            return self.layer.bias * self.layer.biasWeight + self.layer.var * self.layer.biasNoise
        return self.layer.biasWeight

    def forward(self, x):
        if self.layer.ergodic:
            self.resample_noise()
        W    = self.layer.compute_W_current()                         # [nIn, nOut]
        x    = x.to(W.device)
        # Outer product: [..., nIn, 1] * [nIn, nOut] -> [..., nIn, nOut]
        WX   = x.unsqueeze(-1) * W                                   # broadcasts over leading dims
        if self.saturate:
            t     = torch.tanh(WX)
            one_p = 1 + t
            if self.invertible:
                one_m = 1 - t
        else:
            one_p = 1 + WX
            if self.invertible:
                one_m = 1 - WX
        if self.stable:
            one_p = one_p.clamp(min=epsilon)
            if self.invertible:
                one_m = one_m.clamp(min=epsilon)
        y = torch.sum(torch.log(one_p), dim=-2)                      # sum over nIn -> [..., nOut]
        if self.invertible:
            z = torch.sum(torch.log(one_m), dim=-2)                   # [..., nOut]
            # Interleave (y, z) along the object axis:
            # [..., S, nOut] -> [..., S, 2, nOut] -> [..., 2*S, nOut]
            result = torch.stack((y, z), dim=-2).flatten(-3, -2)
        else:
            result = torch.exp(y)
        if self.invertible:
            result = self.layer.forwardBiasInterleaved(result)
        else:
            result = self.layer.forwardBias(result)
        return result

    def reverse(self, yz):
        """Recover x from interleaved (y, z) pairs. Requires invertible=True.

        gamma_j = 0.5*(log y_j - log z_j) = sum_i x_i*w_ji = (x@W)_j.
        x = gamma @ W_inv using the materialized inverse of current W.
        """
        if self.invertible:
            yz = self.layer.reverseBiasInterleaved(yz)
        else:
            yz = self.layer.reverseBias(yz)
        # De-interleave: [..., 2*S, nOut] -> [..., S, 2, nOut] -> y, z each [..., S, nOut]
        y, z = yz.unflatten(-2, (-1, 2)).unbind(-2)
        W_inv = self.layer.compute_Winverse_current()   # [nOut, nIn]
        gamma = 0.5 * (y - z)                           # [..., nOut]
        gamma = gamma.to(W_inv.device)
        x     = gamma @ W_inv
        if self.layer.ergodic:
            self.resample_noise()
        return x

    @staticmethod
    def test():
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayerOld(nInput=nInput, nOutput=nOutput)
        device = next(layer.parameters()).device
        x = torch.randn((nBatch, 6, nInput), device=device)
        layer.set_sigma(0.999)
        y = layer(x)
        assert y.shape == (nBatch, 6, nOutput)
        print(f"Original input: {x}")
        print(f"After PiLayerOld: {y}")

        def check_roundtrip(desc, **kwargs):
            kw = dict(nInput=3, nOutput=6, invertible=True)
            kw.update(kwargs)
            layer = PiLayerOld(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            inputs = [('3D [B,S,nIn]', torch.rand(16, 5, nI, device=device).clamp(min=epsilon)),
                      ('2D [B,nIn]',   torch.rand(16, nI,    device=device).clamp(min=epsilon))]
            for tag, x in inputs:
                layer.set_sigma(0.0)
                y = layer.forward(x)
                x_recon = layer.reverse(y)
                error = torch.norm(x - x_recon) / torch.norm(x)
                assert error < 1e-4, f"{desc} {tag}: reconstruction error {error:.2e}"
            print(f"  {desc}: OK")

        print("Invertible PiLayerOld roundtrip variations:")
        check_roundtrip("naive=T hasBias=T", naive=True,  hasBias=True)
        check_roundtrip("naive=T hasBias=F", naive=True,  hasBias=False)
        check_roundtrip("naive=F hasBias=T", naive=False, hasBias=True)
        check_roundtrip("naive=F hasBias=F", naive=False, hasBias=False)
        check_roundtrip("square nIn=nOut=6", naive=True,  hasBias=False, nInput=6, nOutput=6)
        check_roundtrip("ergodic naive=T hasBias=F", naive=True,  hasBias=False, ergodic=True)
        check_roundtrip("ergodic naive=T hasBias=T", naive=True,  hasBias=True,  ergodic=True)
        check_roundtrip("ergodic naive=F hasBias=T", naive=False, hasBias=True,  ergodic=True)
        print("PiLayerOld tests passed.")

    @staticmethod
    def xorTest():
        X = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=TheDevice.get())
        Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32, device=TheDevice.get())
        nInput, nHidden, nOutput = 2, 3, 1
        pi    = PiLayerOld(nInput, nHidden)
        sigma = SigmaLayer(nHidden, nOutput)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(chain(pi.parameters(), sigma.parameters()), lr=0.01)
        sigma.set_sigma(0.9999); pi.set_sigma(0.9999)
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = criterion(sigma(pi(X)), Y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/1000, MSE: {loss.item():.6f}')


class OldSyntacticLayer(Layer):
    """Recursive derivation stack that learns grammar rule distributions.

    A single shared derivation layer and rule head are applied repeatedly
    at each depth, with a learned depth embedding added to the hidden
    state so the shared weights can specialize by tree level.  This is
    a recursive (weight-tied) architecture — the same transformation
    applies at every level of the derivation, mirroring the recursive
    nature of natural-language syntax.

    **Forward:** receives activation [B, nSymbols], unrolls the shared
    layer for ``max_depth`` steps, predicts a rule distribution at each
    step via Gumbel-softmax, and assembles word tuples.

    **Reverse:** deterministic tree-walk — reads word tuples and
    reconstructs the activation vector by marking referenced positions
    as active.  No learned weights; gradients do not flow through reverse.
    """

    def __init__(self, nInput, nOutput, max_depth=12, hidden_dim=256,
                 grammar=None, tau=1.0):
        super().__init__(nInput, nOutput)
        from Space import Grammar
        from Model import LinearLayer
        self.grammar    = grammar or Grammar()
        self.num_rules  = len(self.grammar)
        self.max_depth  = max_depth
        self.hidden_dim = hidden_dim
        self.tau        = tau

        self.input_proj = LinearLayer(nInput, hidden_dim)
        self.derivation_layer = LinearLayer(hidden_dim, hidden_dim)
        self.rule_head        = LinearLayer(hidden_dim, self.num_rules)
        self.depth_embed = nn.Embedding(max_depth, hidden_dim)
        self.activation_fn = nn.GELU()
        self.layers = [self.input_proj, self.derivation_layer, self.rule_head]

    def forward(self, x):
        B = x.shape[0]
        h = self.input_proj.forward(x)
        h = self.activation_fn(h)

        depth_ids = torch.arange(self.max_depth, device=x.device)
        depth_vecs = self.depth_embed(depth_ids)

        all_logits = []
        all_probs  = []

        for d in range(self.max_depth):
            h = h + depth_vecs[d]
            h = self.derivation_layer.forward(h)
            h = self.activation_fn(h)
            logits = self.rule_head.forward(h)

            if self.training:
                probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            else:
                probs = F.softmax(logits, dim=-1)

            all_logits.append(logits)
            all_probs.append(probs)

        rule_logits     = torch.stack(all_logits, dim=1)
        rule_probs      = torch.stack(all_probs, dim=1)
        predicted_rules = rule_logits.argmax(dim=-1)

        active_positions = self._active_positions(x)
        words = self._generate_derivation(predicted_rules, active_positions)

        return {
            "rule_logits":     rule_logits,
            "rule_probs":      rule_probs,
            "predicted_rules": predicted_rules,
            "words":           words,
        }

    def _active_positions(self, x):
        B = x.shape[0]
        positions = []
        for b in range(B):
            active = torch.nonzero(x[b], as_tuple=False).squeeze(-1)
            positions.append(active.tolist())
        return positions

    def _generate_derivation(self, predicted_rules, active_positions):
        B = predicted_rules.shape[0]
        all_words = []
        for b in range(B):
            rules     = predicted_rules[b].tolist()
            positions = active_positions[b]
            n = len(positions)
            if n == 0:
                continue
            if n == 1:
                all_words.append((b, positions[0], 1))
                continue
            pos_idx = 0
            for rule_id in rules:
                if pos_idx >= n - 1:
                    break
                if self.grammar.arity(rule_id) != 2:
                    rule_id = self.grammar.binary_rules()[0]
                all_words.append((b, positions[pos_idx], rule_id))
                pos_idx += 1
            all_words.append((b, positions[-1], 1))
        return all_words

    def reverse(self, words, nVectors, batch_size):
        from util import TheDevice
        activation = torch.zeros(batch_size, nVectors, device=TheDevice.get())
        for b, v, r in words:
            activation[b, v] = 1.0
        return activation

    def set_tau(self, tau):
        self.tau = tau


# ── Old LogicLayer (fuzzy mereology statics) ─────────────────────────────
# Moved here 2026-04-04.  Replaced by TruthLayer in Model.py.

class OldLogicLayer(Layer):
    """Fuzzy mereology and negation operations on vector sets.

    All methods are static — the class carries no learnable parameters.
    Kept for reference and testing; superseded by TruthLayer
    (truth-store design) in Model.py.
    """

    @staticmethod
    def _pairwise_sq_dists(X, Y):
        x2 = (X * X).sum(dim=-1, keepdim=True)
        y2 = (Y * Y).sum(dim=-1).unsqueeze(1)
        xy = torch.bmm(X, Y.transpose(1, 2))
        d2 = x2 + y2 - 2.0 * xy
        return d2.clamp_min(0.0)

    @staticmethod
    def _expand_sigma(sigma, B, N, device, dtype):
        if sigma is None:
            return torch.ones(B, N, device=device, dtype=dtype)
        if isinstance(sigma, (float, int)):
            return torch.full((B, N), float(sigma), device=device, dtype=dtype)
        if sigma.ndim == 1:
            if sigma.shape[0] != N:
                raise ValueError(f"1D sigma must have shape ({N},), got {tuple(sigma.shape)}")
            return sigma.to(device=device, dtype=dtype).unsqueeze(0).expand(B, N)
        if sigma.ndim == 2:
            if sigma.shape != (B, N):
                raise ValueError(f"2D sigma must have shape ({B}, {N}), got {tuple(sigma.shape)}")
            return sigma.to(device=device, dtype=dtype)
        raise ValueError("sigma must be None, scalar, shape (N,), or shape (B,N)")

    @staticmethod
    def kernel_overlap(X, Y, sigma_x=None, sigma_y=None, eps=1e-8):
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")
        B, N, D = X.shape
        M = Y.shape[1]
        sx = OldLogicLayer._expand_sigma(sigma_x, B, N, X.device, X.dtype)
        sy = OldLogicLayer._expand_sigma(sigma_y, B, M, Y.device, Y.dtype)
        d2 = OldLogicLayer._pairwise_sq_dists(X, Y)
        denom = 2.0 * (sx.unsqueeze(2).square() + sy.unsqueeze(1).square()) + eps
        return torch.exp(-d2 / denom)

    @staticmethod
    def union(X, Y):
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")
        return torch.cat([X, Y], dim=1)

    @staticmethod
    def intersection(X, Y, sigma_x=None, sigma_y=None, topk=None,
                     weight_threshold=None, eps=1e-8):
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")
        B, N, D = X.shape
        M = Y.shape[1]
        sx = OldLogicLayer._expand_sigma(sigma_x, B, N, X.device, X.dtype)
        sy = OldLogicLayer._expand_sigma(sigma_y, B, M, Y.device, Y.dtype)
        Kxy = OldLogicLayer.kernel_overlap(X, Y, sx, sy, eps=eps)
        px = 1.0 / (sx.square().unsqueeze(2) + eps)
        py = 1.0 / (sy.square().unsqueeze(1) + eps)
        denom = px + py
        Xp = X.unsqueeze(2)
        Yp = Y.unsqueeze(1)
        Z = (px.unsqueeze(-1) * Xp + py.unsqueeze(-1) * Yp) / denom.unsqueeze(-1)
        W = Kxy
        if weight_threshold is not None:
            mask = W >= weight_threshold
            W = W * mask
        Z = Z.reshape(B, N * M, D)
        W = W.reshape(B, N * M)
        if topk is not None:
            k = min(topk, N * M)
            vals, idx = torch.topk(W, k=k, dim=1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, D)
            Z = torch.gather(Z, dim=1, index=gather_idx)
            W = vals
        return Z, W

    @staticmethod
    def part(X, Y, sigma_x=None, sigma_y=None, weights_x=None,
             weights_y=None, signed=False, eps=1e-8):
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")
        B, N, D = X.shape
        M = Y.shape[1]
        Kxy = OldLogicLayer.kernel_overlap(X, Y, sigma_x=sigma_x,
                                           sigma_y=sigma_y, eps=eps)
        Kyx = Kxy.transpose(1, 2)
        cover_xy = Kxy.max(dim=2).values
        cover_yx = Kyx.max(dim=2).values
        if weights_x is None:
            p_xy = cover_xy.mean(dim=1)
        else:
            if weights_x.shape != (B, N):
                raise ValueError(f"weights_x must have shape ({B}, {N})")
            wx = weights_x.to(device=X.device, dtype=X.dtype)
            p_xy = (wx * cover_xy).sum(dim=1) / (wx.sum(dim=1) + eps)
        if weights_y is None:
            p_yx = cover_yx.mean(dim=1)
        else:
            if weights_y.shape != (B, M):
                raise ValueError(f"weights_y must have shape ({B}, {M})")
            wy = weights_y.to(device=Y.device, dtype=Y.dtype)
            p_yx = (wy * cover_yx).sum(dim=1) / (wy.sum(dim=1) + eps)
        out = torch.stack([p_xy, p_yx], dim=1)
        if signed:
            out = 2.0 * out - 1.0
        return out

    @staticmethod
    def neg(X):
        return -X

    @staticmethod
    def non(X, alpha=0.0):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        return alpha * X

    @staticmethod
    def symbolize(X, eps=1e-8):
        norms = torch.linalg.norm(X, dim=-1)
        s = 2.0 * norms.mean(dim=1) - 1.0
        return s.clamp(-1.0, 1.0)

    @staticmethod
    def scalar_neg(a):
        return -a

    @staticmethod
    def scalar_non(a, alpha=0.0):
        return (alpha * a).clamp(-1.0, 1.0)

    @staticmethod
    def scalar_union(a, b):
        return torch.maximum(a, b)

    @staticmethod
    def scalar_intersection(a, b):
        return torch.minimum(a, b)

    @staticmethod
    def scalar_part(a, b):
        return (b - a).clamp(-1.0, 1.0)

    @staticmethod
    def test():
        from util import TheDevice
        B, N, M, D = 4, 16, 20, 32
        X = torch.randn(B, N, D, device=TheDevice.get())
        Y = torch.randn(B, M, D, device=TheDevice.get())
        LL = OldLogicLayer(N, M)
        U = LL.union(X, Y)
        Z, W = LL.intersection(X, Y, topk=32)
        P = LL.part(X, Y)
        X_neg = LL.neg(X)
        X_non = LL.non(X, alpha=0.2)
