# Experiment 014: Neural Cellular Automata Gaussians (NCA-GS)

## Date Started
2026-01-28

## Hypothesis

Feed-forward Gaussian decoders produce "blobby" outputs because they predict all parameters in a single forward pass without iterative refinement or local neighborhood awareness. By treating Gaussians as cells in a cellular automaton that update their parameters based on learned local rules over N timesteps, we can achieve emergent global structure from local interactions.

## Theoretical Basis

1. **Neural Cellular Automata** (Mordvintsev et al., 2020): Local update rules can produce complex global patterns through iterative refinement.

2. **Morphogenesis Analogy**: Biological systems create complex 3D structures from local cell-cell interactions. Each "cell" (Gaussian) perceives its neighbors and updates accordingly.

3. **Breaking Symmetry**: Feed-forward networks converge to "safe" average predictions (blobs). Iterative local refinement can break this symmetry and develop sharp features.

4. **Prior Work in This Codebase**:
   - Exp 002: Documented "blobby output" problem persists across hyperparameters
   - Exp 013: Fibonacci sampling (377 points) works mechanically but quality remains poor
   - `tensegrity_loss`: Already computes k-nearest neighbors for regularization

## Approach

Replace direct feed-forward prediction with:

1. **Initialization**: Predict initial Gaussian state from DINOv2 features (like FibonacciPatchDecoder)

2. **NCA Dynamics**: For N timesteps:
   - Each Gaussian finds its k-nearest neighbors (based on position)
   - Concatenate self + neighbor states
   - Pass through perception network
   - Compute update delta via update rule network
   - Apply stochastic masking (NCA stability technique)
   - Update state: `state += step_size * delta`

3. **Output**: Parse final state into Gaussian parameters

## Key Design Choices

- **377 Fibonacci points**: O(N²) neighbor computation tractable (vs 1369 grid points)
- **k=6 neighbors**: Standard NCA neighborhood size
- **16 NCA steps**: Enough iterations for global organization
- **Stochastic masking (p=0.5)**: Random subset updates per step (NCA stability trick)
- **Learnable step size**: Network can adapt update magnitude

## Expected Outcomes

**If hypothesis is correct:**
- Loss should decrease with n_steps > 1 (NCA dynamics contributing)
- Visual output should show less "blobby", more structured Gaussians
- Intermediate steps should show progressive organization

**If hypothesis is wrong:**
- n_steps=1 and n_steps=16 perform similarly (NCA not helping)
- Training may be unstable (updates diverge)
- Could be slower without quality improvement

## Validation Plan

1. **Quick test (<30 min)**: Train 1 epoch on 10 images, verify loss decreases
2. **Step ablation**: Compare n_steps={1, 4, 8, 16}
3. **Visual trajectory**: Render Gaussians at intermediate steps
4. **Full training**: 15 epochs if quick tests pass

## References

- Mordvintsev et al. (2020). "Growing Neural Cellular Automata"
- Growing Neural Cellular Automata: https://distill.pub/2020/growing-ca/
- tensegrity_loss in gaussian_decoder_models.py (existing neighbor computation)
