# Conversion of the swing equation to a system of first-order ODEs

1. Define dummy variables

$$
\eta_k^{1} = \delta_{k}
\\
\eta_k^{2} = \dot{\delta_{k}}
$$

2. Differentiate the dummy variables

$$
\dot{\eta_{k}^{1}} = \dot{\delta_{k}} = \eta_{k}^{2}
\\
\dot{\eta_{k}^{2}} = \ddot{\delta_{k}} 
$$

3. Form a system of first-order ODEs

$$
\mathbf{\dot{\eta}} = \begin{bmatrix} \dot{\eta_{k}^{1}} \\ \dot{\eta_{k}^{2}} \end{bmatrix} = \begin{bmatrix} \dot{\delta_k} \\ \ddot{\delta_k} \end{bmatrix}
$$
