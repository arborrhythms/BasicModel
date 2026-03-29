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
