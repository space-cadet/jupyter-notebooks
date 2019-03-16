
# coding: utf-8

# ## Mathjax custom 
# 
# $ \newcommand{\opexpect}[3]{\langle #1 \vert #2 \vert #3 \rangle} $
# $ \newcommand{\rarrow}{\rightarrow} $
# $ \newcommand{\bra}{\langle} $
# $ \newcommand{\ket}{\rangle} $
# 
# $ \newcommand{\up}{\uparrow} $
# $ \newcommand{\down}{\downarrow} $
# 
# $ \newcommand{\mb}[1]{\mathbf{#1}} $
# $ \newcommand{\mc}[1]{\mathcal{#1}} $
# $ \newcommand{\mbb}[1]{\mathbb{#1}} $
# $ \newcommand{\mf}[1]{\mathfrak{#1}} $
# 
# $ \newcommand{\vect}[1]{\boldsymbol{\mathrm{#1}}} $
# $ \newcommand{\expect}[1]{\langle #1\rangle} $
# 
# $ \newcommand{\innerp}[2]{\langle #1 \vert #2 \rangle} $
# $ \newcommand{\fullbra}[1]{\langle #1 \vert} $
# $ \newcommand{\fullket}[1]{\vert #1 \rangle} $
# $ \newcommand{\supersc}[1]{^{\text{#1}}} $
# $ \newcommand{\subsc}[1]{_{\text{#1}}} $
# $ \newcommand{\sltwoc}{SL(2,\mathbb{C})} $
# $ \newcommand{\sltwoz}{SL(2,\mathbb{Z})} $
# 
# $ \newcommand{\utilde}[1]{\underset{\sim}{#1}} $

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
get_ipython().magic(u'matplotlib inline')


# ## Yang-Baxter Equation
# 
# [Reference](http://arxiv.org/abs/1507.05979)
# 
# Let $V$ be a finite-dimensional Hilbert space, let $R$ be a linear operator on $V \otimes V$. Then $R$ satisfies the constant YBE if:
# 
# $$ (R \otimes \mbb{1})(\mbb{1} \otimes R)(R \otimes \mbb{1}) = (\mbb{1} \otimes R)(R \otimes \mbb{1})(\mbb{1} \otimes R)$$
# 
# In particular, if $V$ is the two-dimensional state space of a qubit or spin-1/2 particle, then $V \otimes V$ is the state space of two qubits. Let $\fullket{0},\fullket{1}$ be the basis states of $V$. Then $V \otimes V$ is spanned by the set:
# 
# $$ \{ \fullket{00}, \fullket{01}, \fullket{10}, \fullket{11} \} $$
# 
# In this case, $R:V \otimes V \rightarrow V \otimes V$, is a linear map on the two qubit state space. An example of such a map is the 

# In[77]:

h2 = hadamard_transform(N=3)
h2


# In[78]:

h2.dims, h2.shape


# In[81]:

v1 = basis(3,0)
v2 = basis(2,1)
v1, v2


# In[82]:

tensor([v1,v2])


# In[122]:

from operator import *


# ### def: `qubitStateToLatex`

# In[103]:

def qubitStateToLatex(state, dim_list ):
    '''Given a QuTiP ket or bra corresponding to the state of an n-site spin chain,
    this function returns the LaTeX string for the state written as a sum over all the
    basis states.
    
    Parameters
    ----------
    state : Qobj
        Bra or Ket of type qutip.Qobj
    
    dim_list : list
        List of dimensions of the systems which the state describes
    
    Returns
    -------
    string : str
        LaTeX formatted string
    
    Examples
    --------
    Given a vector with components $(a,b,c,d)$ in the two-spin Hilbert space,
    returns the string:
        $ a |00> + b |01> + c |10> + d |11> $
    >>> v = tensor([0.5*basis(2,0),0.5*basis(2,1)]) + tensor([0.5*basis(2,1),0.5*basis(2,1)])
    >>> v
    (Quantum object: dims = [[2, 2], [1, 1]], shape = [4, 1], type = ket
     Qobj data =
     [[ 0.  ]
      [ 0.25]
      [ 0.  ]
      [ 0.25]],)
    >>> qubitStateToLatex(v,[2,2])
        $ 0.25 |01> + 0.25 |11> $
    '''
    
    data = 0
    
    # insert check to ensure QuTiP module is imported
    # inset check to ensure operator module is imported
    
    if not isinstance(state,Qobj):
        raise ValueError('First argument must be QuTiP Qobj bra or ket')
    if not state.isket or state.isbra:
        raise ValueError('First argument must be a QuTiP ket or bra')

    if not isinstance(dim_list,list):
        raise ValueError('Second argument must be a list')
        
    for i in dim_list:
        if not (isinstance(i,int) and (i>0)):
            raise ValueError('Second argument of positive, non-zero integers')
    
    # Determine the dimension of the tensor product space
    tensor_dim = map(mul,dim_list)
    
    is_bra = False
    is_ket = False
    
    if state.isket:
        is_ket = True
    elif state.isbra:
        is_bra = True
        state = state.conj()
    
    vect_dim = state.shape[0]
    
    # The input state must live in the tensor product space of particles with Hilbert spaces of dimensions
    # given in dim_list. Thus vect_dim - the dimension of the Hilbert space to which input state belongs
    # must be equal to tensor_dim.
    
    if vect_dim != tensor_dim:
        raise ValueError('Dimension of input state is not compatible with dimensions given in dim_list')
    
    
    
    
    if state.shape == [1,1]:
        data = state.full()
        return data[0,0]
        
    


# In[121]:


map(mul,[1,2,3])


# ### def: `qubitSequenceToState`

# In[44]:

v = hadamard_transform(2)*ket('00')
v


# In[120]:

v.shape[0]


# In[108]:

a = Qobj(1.5)
a


# In[110]:

a.full()


# In[118]:

_110[0,0]


# In[119]:

type(_118)


# In[46]:

get_ipython().magic(u'pinfo v.matrix_element')


# In[97]:

[v1,v2,v3,v4] = [ket('00'), ket('01'), ket('10'), ket('11')]
v1, v2, v3, v4


# In[98]:

v1.trans()


# In[99]:

v1.shape


# In[102]:

tensor([basis(2,0),basis(1,0)])


# In[60]:

# this is nothing more than the function ket() which is already defined in QuTiP
def qubitSequenceToState(x = '01'):
    ''' Given a string of length N, consisting only of 0s and 1s, constructs and returns
    the corresponding quantum state.

    For e.g., if SequenceToState('101') returns
    tensor([basis(2,1),basis(2,0),basis(2,1)])
    '''
    
    if not isinstance(x,str):
        raise ValueError('Function takes exactly one argument of type string')
    else:
        n = len(x)
        if (x.count('0') + x.count('1')) != n:
            raise ValueError('Argument should be string consisting only of 0s and 1s')
    
    op_list = []
    
    for i in range(n):
        op_list.append(basis(2,int(x[i])))
        
    return tensor(op_list)
        


# In[35]:

qubitSequenceToState()


# In[32]:

ket('00')

