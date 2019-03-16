# QuTiP Utility Functions
# Dec 24, 2015
# Deepak Vaid

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

try:
    from qutip import *
except:
    raise NameError('QuTiP is not installed')

def identityList(N = 1,dims = 2):
    '''Returns a list of N identity operators for a N site spin-system with a
    Hilber space at each state of dimensionality dims 
    '''

    iden = identity(dims)
    
    iden_list = []

    [iden_list.append(iden) for i in range(N)]
    
    return iden_list

def posOp2d(oper, pos = (0,0), latt_size = (1,1) ):
    '''Returns QuTiP operator in position representation defined on site (i,j) passed as argument pos, of 2d lattice
    of size specified in tuple latt_size.
    '''
    
    if not isinstance(oper, Qobj):
        raise TypeError('oper must of type qutip.Qobj')
    
    if not oper.isoper:
        raise ValueError('oper must be a qutip operator')
        
    shape = oper.shape

    if shape[0] == shape[1]:
        dims = shape[0]
    else:
        raise ValueError('oper must be a square matrix')
        
    if (not isinstance(pos, tuple)) or (not isintance(latt_size, tuple)):
        raise TypeError('pos and latt_size must be tuples') 
    

def posOperatorN(oper, i = 0, N = 1):
    '''Returns the operator given by oper, in the position representation, for the i^th site of an N site spin-chain
    with a Hilbert space of dimensionality dims at each site'''
    
    if not isinstance(oper, Qobj):
        raise TypeError('oper must of type qutip.Qobj')
    
    if not oper.isoper:
        raise ValueError('oper must be a qutip operator')
    
    shape = oper.shape

    if shape[0] == shape[1]:
        dims = shape[0]
    else:
        raise ValueError('oper must be a square matrix')
    
    if oper == identity(oper.shape[0]):
        return tensor(identityList(N,oper.shape[0]))
    else:
        iden_list = identityList(N, oper.shape[0])
        iden_list[i] = oper
        return tensor(iden_list)
    
def posCreationOpN(i=0, N=10):
    '''Returns the creation operator in the position representation for the i^th site of an N site spin-chain
    with a Hilbert space of dimensionality dims at each site'''
    
    return posOperatorN(create(2),i,N)

def posDestructionOpN(i=0, N=10):
    '''Returns the destruction operator in the position representation for the i^th site of an N site spin-chain
    with a Hilbert space of dimensionality dims at each site'''
    
    return posOperatorN(destroy(2),i,N)


# In[8]:

def posOperHamiltonian(oper, N = 10, coefs = [1]*10, dims=2):
    ''' Returns the Hamiltonian, in position representation, given by the sum of oper acting on each site
    with a weight given by the values in coefs
    '''
    
    if not isinstance(oper, Qobj):
        raise ValueError('oper must be of type Qobj')
    else:
        if not oper.isoper:
            raise ValueError('oper must be an operator')
    
    H = 0
    
    for i in range(N):
        op_list = identityList(N, dims)
        op_list[i] = oper
        H += coefs[i]*tensor(op_list)
    
    return H

def posToMomentumOpN(oper, k = 0, N = 1):
    '''Returns the momentum space represenation of the operator given by oper for the k^th momentum
    of an N site spin-chain with a Hilbert space of dimensionality dims at each site'''
    
    momOp = tensor(identityList(N,oper.shape[0]))
    invrtn = 1/np.sqrt(N)
    
#    type(invrtn)

    for i in range(N):
        momOp += invrtn * np.exp(1j*i*k/(2*np.pi*N)) * posOperatorN(oper,i,N)
        
    return momOp


# In[11]:

def momCreationOpN(k = 0, N = 1):
    '''Returns the momentum space represenation of the creation operator for the k^th momentum
    of an N site spin-chain with a Hilbert space of dimensionality dims at each site'''
        
    return posToMomentumOpN(create(2),k,N)


# In[12]:

def momDestructionOpN(k = 0, N = 1):
    '''Returns the momentum space represenation of the operator given by oper for the k^th momentum
    of an N site spin-chain with a Hilbert space of dimensionality dims at each site'''
    
    return qutip.dag(momCreationOpN(k,N))


# In[13]:

def matrixPosToMom(N = 10):
    ''' Returns a QuTiP object corresponding to a matrix whose elements are:
    
    U_{k,l} = e^{-i*k*l}
    
    '''
    
    matrix = np.zeros([N,N],dtype=complex)
    
    for k in range(N):
        for l in range(N):
            matrix[k][l] = np.exp(1j*k*l)
    
    return Qobj(matrix)

def hamiltonianHubbard(N = 10, t = 1, U = 1, mu = 0, periodic = True, shift = False, dims = 2):
    '''Returns operator corresponding to Hubbard Hamiltonian on N sites.
    Default value of N is 10. t, U and mu are the kinetic energy, potential energy and
    chemical potential respectively.
    If shift is False then first version of Hamiltonian is returned, else the second
    version (where the chemical potential is shifted $\mu \rightarrow \mu - U/2$) is
    returned.
    dims is the dimension of the Hilbert space for each electron. Default is 2'''
    
    # two sets of creation/destruction operators, labeled A and B.
    
    destroyA_list = []
    createA_list = []

    destroyB_list = []
    createB_list = []
    
    cOp = create(dims)
    dOp = destroy(dims)
    nOp = cOp * dOp

    nA_list = []
    nB_list = []
    
    idOp = identity(dims)

    idOp_list = []

    [idOp_list.append(idOp) for i in range(N)]
    
    superid = tensor(idOp_list) # identity operator for whole system
    
    H = 0

    for i in range(N):
        # Create list containing creation/destruction/number operators for each site

        createA_list.append(posOperatorN(cOp,N=N,i=i))
        createB_list.append(posOperatorN(cOp,N=N,i=i))

        destroyA_list.append(posOperatorN(dOp,N=N,i=i))
        destroyB_list.append(posOperatorN(dOp,N=N,i=i))

        nA_list.append(posOperatorN(nOp,N=N,i=i))
        nB_list.append(posOperatorN(nOp,N=N,i=i))
        
    if periodic == True:
        for i in range(N):
            H += - t * (createA_list[i%N] * destroyA_list[(i+1)%N] + createB_list[i%N] * destroyB_list[(i+1)%N])
    else:
        for i in range(N-1):
            H += - t * (createA_list[i] * destroyA_list[i+1] + createB_list[i] * destroyB_list[i+1])

    for i in range(N):
        H += - mu * (nA_list[i] + nB_list[i])
        if shift == True:
            H += U * (nA_list[i] - 0.5 * superid) * (nB_list[i] - 0.5 * superid)
        else:
            H += U * nA_list[i] * nB_list[i]
    
    return H

def hamiltonianIsing(N = 10, jcoefs = [], periodic = True, spin=0.5):
    '''Returns operator corresponding to Ising Hamiltonian for give spin on N sites.
    Default value of N is 10. jcoef is the coupling strength. Default is -1 for
    ferromagnetic interaction.
    Default value of spin is 0.5
    '''
    
    op_list = []
    
    jz = jmat(spin,'z')
    
    dimj = 2*spin + 1
    
    H = 0
    
    idlist = identityList(N, dimj)
    
    if len(jcoefs) == 0:
        jcoefs = [-1]*N
    
    for i in range(N):
        # Create list containing spin-z operators for each site:
        
        op_list.append(posOperatorN(jz,i=i,N=N))
    
    if periodic == True:
        for i in range(N):
            H += jcoefs[i%N]*op_list[i%N]*op_list[(i+1)%N]
    else:
        for i in range(N-1):
            H += jcoefs[i]*op_list[i]*op_list[i+1]
            
    return H

def hamiltonianHeisenberg(N = 5, J = 1, periodic = True):
    '''Returns operator corresponding to the Heisenberg 1D spin-chain on N sites.
    $$ H = - J \sum_{i=1}^N S_n \cdot S_{n+1} $$
    where $ S_n (S_x, S_y, S_z) $ is the spin-operator acting on the n^{th} site.
    
    H can be written in terms of spin-flip operators $S^+, S^-$ as:
    
    $$ H = -J \sum_{i=1}^N \left[ \frac{1}{2} (S_n^+ S_{n+1}^- + S_n^- S_{n+1}^+) + S_n^z S_{n+1}^z \right] $$
    
    '''
    
    spinp_list = []
    spinm_list = []
    spinz_list = []
    
    opSpinP = sigmap()
    opSpinM = sigmam()
    opSpinZ = sigmaz()
    
    idOp = identity(2)

    idOp_list = []
    
    [idOp_list.append(idOp) for i in range(N)]
    
    superid = tensor(idOp_list) # identity operator for whole system
    
    H = 0

    for i in range(N):
        # Create list containing creation/destruction/number operators for each site

        spinp_list.append(posOperatorN(opSpinP,N=N,i=i))
        spinm_list.append(posOperatorN(opSpinM,N=N,i=i))
        spinz_list.append(posOperatorN(opSpinZ,N=N,i=i))
    
    if periodic == True:
        for i in range(N):
            H += - J * ( 0.5*(spinp_list[i%N] * spinp_list[(i+1)%N] + spinm_list[i%N] * spinm_list[(i+1)%N])                     + spinz_list[i%N] * spinz_list[(i+1)%N] )
    else:
        for i in range(N-1):
            H += - J * ( 0.5*(spinp_list[i] * spinp_list[i+1] + spinm_list[i] * spinm_list[i+1])                     + spinz_list[i] * spinz_list[i+1] )
    
    return H

def jordanWignerDestroyI(i = 0, N = 1):
    ''' Returns the fermionic annihilation operator for the i^th site of a N site spin-chain:
    a_i = Z_1 x Z_2 ... x Z_{i-1} x sigma_i
    where Z_i is the Pauli sigma_z matrix acting on the i^th site and sigma_i is the density
    matrix acting on the i^th qubit, given by:
    sigma_i = id_1 x id_2 ... x id_{i-1} x |0><1| x id_{i+1} ... x id_n
    where id_i is the identity operator
    
    Reference: Nielsen, Fermionic CCR and Jordan Wigner Transformation
    '''
    
    # create zop, assign to it identity operator for N site system
    zop = tensor([qeye(2)]*N)
    
    # for k in (0..i-1), create Z_k operator for N-site chain
    # zop is product of Z_1 * Z_2 ... * Z_{i-1}
    
    for k in range(i):
        zop *= posOperatorN(sigmaz(),i = k, N = N)
    
    # create single qubit density matrix |0><1|
    
    sigmai = ket([0])*bra([1])
    
    # create list of N single-site identity operators
    
    op_list = identityList(N = N)

    # assign single qubit density matrix |0><1| to i^th item of above list
    
    op_list[i] = sigmai
    
    # take the tensor product of resulting list to obtain operator for density matrix |0><1|
    # acting on i^th qubit of N-site chain.
    
    sigmaop = tensor(op_list)
    
    # return zop*sigmaop = Z_1 x Z_2 .. x Z_{i-1} x sigma_i, which is fermionic annihilation
    # operator for i^th site of N-site spin-chain.
    
    return zop * sigmaop

def offDiagBelow(N,a=1):
    """
    Returns a NxN array with the elements below the diagonal set to a
    """
    return a*(np.tri(N,N,-1) - np.tri(N,N,-2))

def offDiagAbove(N,a=1):
    """
    Returns a NxN array with the elements above the diagonal set to a
    """
    return a*(np.tri(N,N,1) - np.tri(N,N))

class Hamiltonian(Qobj):
    
    _hamTypes = ['NumberOp', 'Ising', 'Heisenberg','Hubbard']
    _hamType = ''
    _maxSites = 100
    _numSites = 1
    _dims = 2
    _label = None
    _data = None
    
    _hamiltonian = Qobj()
    _eigenenergies = []
    _eigenstates = []
    _isHermitian = True
    
    def __init__(self, label=None, dims=2, isHermitian=True, numSites=1, hamType=None,data=None):
        
        if numSites<1 or not isinstance(numSites,int):
            raise ValueError('numSites must be an integer greater than or equal to 1')
        if numSites>self._maxSites:
            raise ValueError('numSites cannot be greater than ' + str(self._maxSites))
        else:
            self._numSites = numSites
        
        if label!=None and isinstance(label, str):
            self._label = label
            
        if data!=None:
            self._data = data

        self._isHermitian = isHermitian
        
        if hamType != None:
            if hamType not in self._hamTypes:
                from string import join
                raise ValueError('hamType must be one of ' + join(self._hamTypes,', '))
            else:
                self._hamType = hamType
                self.createHamiltonian()
        else:
            self._hamiltonian = Qobj()
        
        if dims < 2 or not isinstance(dims, int):
            raise ValueError('dim must be an integer greater than or equal to 2')
        else:
            self._dims = dims
            
        Qobj.__init__(self._hamiltonian)
        
        return

    def createHamiltonian(self):
        
        if self._hamType == 'Ising':
            
            self._hamiltonian = hamiltonianIsing(self._numSites,self._data['jcoefs'],self._data['spin'])
        
        elif self._hamType == 'Hubbard':
            
            self._hamiltonian = hamiltonianHubbard(self._numSites, self._data['t'], self._data['U'], self._data['mu'], self._data['shift'], self._dims)
            
        elif self._hamType == 'NumberOp':
            
            numOp = create(self._dims)*destroy(self._dims)
            
            self._hamiltonian = posOperHamiltonian(numOp, self._numSites, self._data['coefs'], self._dims)
            
        elif self._hamType == 'Heisenberg':
            
            self._hamiltonian = hamiltonianHeisenberg()
            
        return
            
    @property
    def hermitian(self):
        return self._isHermitian
    
    @hermitian.setter
    def hermitian(self, value):
        if isinstance(value, bool):
            self._isHermitian = value
        else:
            raise ValueError('hermitian must be a boolean data type')


# ## Vacuum State, Basis States

def stateVacuum(N = 10):
    ''' Returns a QuTiP object representing the vacuum state for a spin-chain with N sites '''
    
    state = []
    
    for i in range(N):
        state.append(basis(2,0))
    
    return tensor(state)


def stateUpK(k = 0, N = 10):
    ''' Returns a QuTiP object representating a state with the k^th spin pointing up and the rest pointing
    down for a N site spin-chain. '''
    
    state = []
    
    for i in range(N):
        state.append(basis(2,0))
        
    state[k] = basis(2,1)
    
    return tensor(state)


def stateDownK(k = 0, N = 10):
    ''' Returns a QuTiP object representating a state with the k^th spin pointing down and the rest pointing
    up for a N site spin-chain. '''
    
    state = []
    
    for i in range(N):
        state.append(basis(2,1))
        
    state[k] = basis(2,0)
    
    return tensor(state)

def stateListDown(N = 10):
    ''' Returns a list of QuTiP objects, whose elements are the mutually orthogonal states, with mostly
    down spins, of a spin-chain with N sites. '''
    
    stateList = []
    
    for i in range(N):
        stateList.append(stateDownK(i,N))
        
    return stateList


def stateListUp(N = 10):
    ''' Returns a list of QuTiP objects, whose elements are the mutually orthogonal states, with mostly
    up spins, of a spin-chain with N sites. '''
    
    stateList = []
    
    for i in range(N):
        stateList.append(stateUpK(i,N))
        
    return stateList
