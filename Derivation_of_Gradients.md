# Sobolev gradient flows
## Framework
Given the GPE:
$$
-\Delta z + V z+ \beta |z|^2 z = \lambda z \qquad or \qquad -\frac{1}{2}\Delta z + V z+ \beta |z|^2 z = \lambda z
$$
It's energy is:
$$
E(v) = \frac{1}{2} \int_D |\nabla v|^2 + V |v|^2 + \frac{\beta}{2} |v|^4 dx \qquad or \qquad E(v) = \frac{1}{2} \int_D \frac{1}{2}|\nabla v|^2 + V |v|^2 + \frac{\beta}{2} |v|^4 dx
$$
Its Frechet derivative:
$$
< E'(v), w> = \int_D \nabla v \nabla w + V v w + \beta |v|^2 v w dx \qquad or \qquad < E'(v), w> = \int_D \frac{1}{2}\nabla v \nabla w + V v w + \beta |v|^2 v w dx
$$ 
The eigenvalue associated is derived as $\lambda^* = 2 E(z^*) + \frac{\beta}{2} \|z^*\|^4_{L^4}$
The ground state is defined as:
$$
E(z_{GS}) = \inf \{ E(z) | z \in H^1_0(D) \ and \ \|z\|_{L^2} = 1\}
$$

## Projected Gradient flow
Select X such that $H_0^1(D) \sub X \sub L^2(D)$, denote by $\nabla_X E(z) : H_0^1(D) \rightarrow X$ the Riesz rapresentative of $E'(z)$:
$$
(\nabla_X E(z),v)_X = <E'(z),v> \qquad \forall v \in X
$$
Impose the constrain $\|z\| = 1$ and define $T_{z,X} = \{v \in X | (v,z)_{L^2}= 0\}$:
$$
(P_{z,X}(v),\psi)_X = (v, \psi)_X \qquad \forall v \in T_{z,X}
$$
and in term of Riesz rapresentative:
$$
P_{z,X}(v) = v - \frac{(z,v)_{L^2}}{(z,R_X(z))_{L^2}} R_X(z)
$$
where $R_X(z) \in X$ is the Reisz rapresentative of $z$ in $X$. So the final gradient flow equation is:
$$
z'(t) = - (P_{z,X} \circ \nabla_X E)(z(t))
$$

## Choices of $X$
- ### $X = L^2(D)$ and semi-implicit:
    In this case $R_{L^2}(z) = z$, $P_{z,L^2}(v) = v - \frac{(z,v)_{L^2}}{(z,z)_{L^2}}z$:
    $$
    \nabla_{L^2} E(z) = - \Delta z + V z + \beta \|z\|^2 z \qquad or \qquad \nabla_{L^2} E(z) = - \frac{1}{2}\Delta z + V z + \beta \|z\|^2 z
    $$
    the putting all togheter and hiding the part $\frac{(z,v)_{L^2}}{(z,z)_{L^2}}z$ into the normalization, discretize with backward Euler we get:
    $$
    \tilde z^{n+1} = z^n - \tau \nabla_{L^2} E(z^{n+1})
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$
    so apply the weak form to the equation:
    $$
    \int_D \tilde{z}^{n+1}v dx + \tau \int_D \nabla \tilde{z}^{n+1} \nabla v +V \tilde{z}^{n+1} v + \beta |z^n|^2 \tilde{z}^{n+1} v dx = \int_D u^nv dx
    \qquad or \qquad \int_D \tilde{z}^{n+1}v dx + \tau \int_D \frac{1}{2} \nabla \tilde{z}^{n+1} \nabla v +V \tilde{z}^{n+1} v + \beta |z^n|^2 \tilde{z}^{n+1} v dx = \int_D u^nv dx
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$
- ### $X = L^2(D)$ and explicit:
    In this case $R_{L^2}(z) = z$, $P_{z,L^2}(v) = v - \frac{(z,v)_{L^2}}{(z,z)_{L^2}}z$:
    $$
    \nabla_{L^2} E(z) = - \Delta z + V z + \beta \|z\|^2 z \qquad or \qquad \nabla_{L^2} E(z) = - \frac{1}{2}\Delta z + V z + \beta \|z\|^2 z
    $$
    So discretizing fully explicit:
    $$
    \tilde z^{n+1} = z^n - \tau \nabla_{L^2} E(z^{n})
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$
    so apply the weak form to the equation:
    $$
    \int_D \tilde{z}^{n+1} v dx = \int_D z^n v dx - \tau \int_D \nabla z^n \nabla v + Vz^nv + \beta |z^n|^2 z^n v dx + \tau \frac{\int_D \nabla z^n \nabla z^n + Vz^n z^n + \beta |z^n|^2 z^n z^n dx}{\|z\|^2} \int_D z^n v dx
    \\
    or
    \\
    \int_D \tilde{z}^{n+1} v dx = \int_D z^n v dx - \tau \int_D \frac{1}{2}\nabla z^n \nabla v + Vz^nv + \beta |z^n|^2 z^n v dx + \tau \frac{\int_D \frac{1}{2}\nabla z^n \nabla z^n + Vz^n z^n + \beta |z^n|^2 z^n z^n dx}{\|z\|^2} \int_D z^n v dx

    \\


    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$  

- ### $X = H^1(D)$ and inner product $(.,.)_{H^1} = (\nabla ., \nabla .)_{L^2}$:
    In this case:
    $$
    (R_{H^1}(z),v)_{H^1} = (\nabla R_{H^1}(z),\nabla v)_{L^2} = (z,v)_{L^2} \qquad \forall v \in H^1_0(D)
    \\
    (\nabla E_{H^1}(z),v)_{H^1} = (\nabla \nabla E_{H^1}(z),\nabla v)_{L^2} = \int_D \nabla v \nabla w + V v w + \beta |v|^2 v w dx \qquad or \qquad (\nabla \nabla E_{H^1}(z),\nabla v)_{L^2} = \int_D \frac{1}{2}\nabla v \nabla w + V v w + \beta |v|^2 v w dx 
    $$
    So it's needed to solve 2 previous subproblems and then:
    $$
    \tilde z^{n+1} = z^n - \tau \nabla_{H^1} E(z^{n}) - \tau \frac{(\nabla_{H^1}E(z^n),z^n)_{L^2}}{(R_{H^1}(z^n),z^n)_{L^2}} R_{H^1}(z^n)
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$
    so it's possible to assign pointwise(?)
    <!-- $$
    \int_D \tilde{z}^{n+1}v dx = \int_D z^n v dx - \tau \int_D \nabla E_{H^1}(z^n) v dx + \tau \frac{\int_D \nabla E_{H^1}(z^n)z^n*dx}{\int_D R_{H^1}(z^n) z^n dx} \int_D R_{H^1}(z^n) v dx
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$ -->
- ### $X = H^1_0(D) $ and inner product $a_0(.,.) = \int_D \frac{1}{2}\nabla . \nabla . + V . .dx$:
    In this case Reisz:
    $$
    a_0(R_{a_0}(z),w) = \int_D \nabla R_{a_0}(z) \nabla w + V R_{a_0}(z) w dx = (z,w)_{L^2} = \int_D zw dx \qquad \forall w \in H^1_0(D)
    \\
    or
    \\
    a_0(R_{a_0}(z),w) = \int_D \frac{1}{2}\nabla R_{a_0}(z) \nabla w + V R_{a_0}(z) w dx = (z,w)_{L^2} = \int_D zw dx \qquad \forall w \in H^1_0(D)
    $$
    while the gradient:
    $$
    a_0(\nabla_{a_0}E(z), v) = \int_D \nabla \nabla_{a_0}E(z) \nabla v + V \nabla_{a_0}E(z) v dx = < E'(v), w> = \int_D \nabla v \nabla w + V v w + \beta |v|^2 v w dx 
    \\ 
    or 
    \\
     a_0(\nabla_{a_0}E(z), v) = \int_D \frac{1}{2}\nabla \nabla_{a_0}E(z) \nabla v + V \nabla_{a_0}E(z) v dx = < E'(v), w> = \int_D \frac{1}{2}\nabla v \nabla w + V v w + \beta |v|^2 v w dx
    $$
    so the gradient is:
    $$
    \nabla _{a_0} E(z) = z + R_{a_0}(\beta |z|^2 z)
    $$

    so discretize the equation with forward Euler:
    $$
    \tilde z^{n+1} =z^n - \tau z^n - \tau R_{a_0}(\beta |z|^2z) + \tau \frac{(z^n + R_{a_0}(\beta |z^n|^2 z^n), z^n)_{L^2}}{(R_{a_0}(z^n),z^n)_{L^2}}R_{a_0}(z^n)
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$
    this assignment could be done pointwise(?).
    <!-- the iteration is:
    $$
    \int_D \tilde{z}^{n+1} w dx = \int_D \tilde{z}^{n} w dx - \tau * \int_D (z^n + R_{a_0}(\beta |z^n|^2 z^n))w dx + \tau \frac{\int_D (z^n + R_{a_0}(\beta |z^n|^2 z^n))z^n dx}{\int_D R_{a_0}(z^n)z^n dx} \int_D R_{a_0}(z^n)wdx
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$  -->
- ### $ X = H^1_0(D) $ and inner product $a_z(.,.) = \int_D \frac{1}{2} \nabla . \nabla . + V . . + \beta |z|^2 . . dx$:
    In this case the Riesz projectionis given by:
    $$
    a_z(R_{a_z}(z),w) = \int_D \nabla R_{a_z}(z) \nabla w + V R_{a_z}(z) w + \beta |z|^2 * R_{a_z}(z) w dx = (z,w)_{L^2} = \int_D zw dx \qquad \forall w \in H^1_0(D)
    \\
    or
    \\
    a_z(R_{a_z}(z),w) = \int_D \frac{1}{2}\nabla R_{a_z}(z) \nabla w + V R_{a_z}(z) w + \beta |z|^2 * R_{a_z}(z) w  dx = (z,w)_{L^2} = \int_D zw dx \qquad \forall w \in H^1_0(D)
    $$    
    The gradient becomes simply:
    $$
    \nabla_{a_z} E(z) = z
    $$
    So the final formula:
    $$
    \tilde z^{n+1} = z^n - \tau z^n + \tau \frac{(z^n,z^n)_{L^2}}{(z^n,R_{a_z(z^n)})_{L^2}}R_{a_z}(z^n)
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$
    also pointwise(?)
    <!-- $$
    \int_D \tilde{z}^{n+1} w dx = \int_D z^{n} w dx - \tau  \int_D  z^{n} w dx + \tau \frac{\int_D z^n z^n dx}{\int_D R_{a_z}(z^n)z^n dx} \int_D R_{a_z}(z^n) w dx
    \\
    z^{n+1} = \frac{\tilde{z}^{n+1}}{\|\tilde{z}^{n+1}\|_{L^2}}
    $$ -->


TODO:
- [ x ]CONTROLLARE IL CODICE 
- [ x ]PAPER CITATI ED ALTRI ESPERIMETNI NUMERICI
- [ x ]CAMBIARE H E VEDERE LA CONVERGENZA

- [ x ]TANTE PRIME ITERATE CON VARI h
- [ x ]VERO GRADINETE L2 (BAO 12) CON MASS LAMPING FORSE

- [  ] controlalre quadrature per la non-linearità 