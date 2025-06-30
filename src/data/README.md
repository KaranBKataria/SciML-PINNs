# Conversion of the swing equation to a system of first-order ODEs

## Swing equation

### Without PI controller

$$
\ddot{\delta_k} = \frac{1}{m_k} \Bigg[ P_{k}^{m} - \sum_{j \ne k} B_{kj}V_{k}V_{j}\mathrm{sin}(\delta_k - \delta_j) - d_k \dot{\delta_k} \Bigg] = f(\delta_k, \dot{\delta_k})
$$

### With PI controller

$$
\ddot{\delta_k} = \frac{1}{m_k} \Bigg[ C_{k}^{I} \delta_k - \sum_{j \ne k} B_{kj}V_{k}V_{j}\mathrm{sin}(\delta_k - \delta_j) - (d_k - C_k^{p}) \dot{\delta_k} \Bigg] = f(\delta_k, \dot{\delta_k})
$$

1. Define dummy variables

$$
\eta_k^{(1)} = \delta_{k}
$$

$$
\eta_k^{(2)} = \dot{\delta_{k}}
$$

2. Differentiate the dummy variables

$$
\dot{\eta_{k}^{(1)}} = \dot{\delta_{k}} = \eta_{k}^{(2)}
$$

$$
\dot{\eta_{k}^{(2)}} = \ddot{\delta_{k}} 
$$

3. Form a system of first-order ODEs

$$
\frac{\mathrm{d}}{\mathrm{d}t} \mathbf{\eta} = 
\begin{bmatrix}
\dot{\eta_{k}^{(1)}} \\ 
\dot{\eta_{k}^{(2)}} 
\end{bmatrix} = 
\begin{bmatrix} 
\dot{\delta_k} \\ 
\ddot{\delta_k} 
\end{bmatrix}
$$
