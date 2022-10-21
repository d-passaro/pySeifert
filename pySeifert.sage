r"""
Companion SageMath package to: 3-manifolds and VOA Characters [2201.04640].

We provide a hands on SageMath package developed in combination with [1] to test claims and provide examples. We provide in particular a class: ``Seifert'' representing a Seifert manifold, which can be used to compute a large amount of data, including: its plumbing matrix, its lattice dilation factor, zhat invariant integrands, ...

EXAMPLES::


AUTORS:
    Miranda C. N. Cheng,
    Sungbong Chun,
    Boris Feigin,
    Francesca Ferrari,
    Sergei Gukov,
    Sarah M. Harrison,
    Davide Passaro (2022-05-06): initial version

"""

from sage.all_cmdline import *   # import sage library
import numpy as np
import itertools


_cartan_matrices = dict()
def cartan_matrix(type_rank):
    """
    Compute the Cartan matrix of given Lie group.

    INPUT:
    -   ``type_rank'' - [str,int]; Lie group identifier

    EXAMPLES::
        sage: cartan_matrix(["A",2])
        [ 2 -1]
        [-1  2]
        sage: cartan_matrix(["E",8])
        [ 2  0 -1  0  0  0  0  0]
        [ 0  2  0 -1  0  0  0  0]
        [-1  0  2 -1  0  0  0  0]
        [ 0 -1 -1  2 -1  0  0  0]
        [ 0  0  0 -1  2 -1  0  0]
        [ 0  0  0  0 -1  2 -1  0]
        [ 0  0  0  0  0 -1  2 -1]
        [ 0  0  0  0  0  0 -1  2]

    COMMENTS:
    Cartan matrices that are computed are stored in a dictionary _cartan_matrices
    for easy and quick access.
    """
    if str(type_rank) not in _cartan_matrices.keys():
        _cartan_matrices[str(type_rank)] = CartanMatrix(type_rank)
    return _cartan_matrices[str(type_rank)]


_weyl_groups = dict()
def weyl_group(type_rank):
    """
    Compute the Weyl group of a given Lie group.

    INPUT:
    -   ``type_rank'' - [str,int]; Lie group identifier

    Examples::    
        sage: weyl_group(["A",2])
        [
        [1 0]  [-1  1]  [ 1  0]  [ 0 -1]  [-1  1]  [ 0 -1]
        [0 1], [ 0  1], [ 1 -1], [ 1 -1], [-1  0], [-1  0]
        ]
        sage: weyl_group(["A",3])
        [
        [1 0 0]  [-1  1  0]  [ 1  0  0]  [ 1  0  0]  [ 0 -1  1]  [-1  1  0]
        [0 1 0]  [ 0  1  0]  [ 1 -1  1]  [ 0  1  0]  [ 1 -1  1]  [-1  0  1]
        [0 0 1], [ 0  0  1], [ 0  0  1], [ 0  1 -1], [ 0  0  1], [ 0  0  1],

        [ 1  0  0]  [-1  1  0]  [ 1  0  0]  [ 0 -1  1]  [ 0  0 -1]  [-1  1  0]
        [ 1  0 -1]  [ 0  1  0]  [ 1 -1  1]  [-1  0  1]  [ 1  0 -1]  [-1  1 -1]
        [ 0  1 -1], [ 0  1 -1], [ 1 -1  0], [ 0  0  1], [ 0  1 -1], [ 0  1 -1],

        [ 1  0  0]  [ 0 -1  1]  [-1  1  0]  [ 0  0 -1]  [ 0  0 -1]  [ 0 -1  1]
        [ 1  0 -1]  [ 1 -1  1]  [-1  0  1]  [-1  1 -1]  [ 1  0 -1]  [ 0 -1  0]
        [ 1 -1  0], [ 1 -1  0], [-1  0  0], [ 0  1 -1], [ 1 -1  0], [ 1 -1  0],

        [-1  1  0]  [ 0 -1  1]  [ 0  0 -1]  [ 0  0 -1]  [ 0 -1  1]  [ 0  0 -1]
        [-1  1 -1]  [-1  0  1]  [ 0 -1  0]  [-1  1 -1]  [ 0 -1  0]  [ 0 -1  0]
        [-1  0  0], [-1  0  0], [ 1 -1  0], [-1  0  0], [-1  0  0], [-1  0  0]
    
    COMMENTS:
    Weyl groups that are computed are stored in a dictionary _weyl_groups 
    for easy and quick access.
    """
    if str(type_rank) not in _weyl_groups.keys():
        _weyl_groups[str(type_rank)] = [matrix(g) for g in WeylGroup(
            type_rank).canonical_representation().list()]
    return _weyl_groups[str(type_rank)]


_weyl_vectors = dict()
def weyl_vector(type_rank):
    """
    Compute the Weyl vector of a given Lie group.

    INPUT:
    -   ``type_rank'' - [str,int]; Lie group identifier

    Examples::    
        sage: weyl_vector(["A",2])
        (1,1)
        sage: weyl_vector(["D",4])
        (3,5,3,3)
    
    COMMENTS:
    Weyl vectors that are computed are stored in a dictionary _weyl_groups 
    for easy and quick access.
    """
    if str(type_rank) not in _weyl_vectors.keys():
        WG = WeylGroup(type_rank).canonical_representation()
        _weyl_vectors[str(type_rank)] = 1/2*sum(WG.positive_roots())
    return _weyl_vectors[str(type_rank)]

_weyl_lengths = dict()
def weyl_lengths(type_rank):
    """
    Return weyl lengths of elements of the weyl group

    Inputs:
    -   ``type_rank`` - [str,int]; Lie group identifier

    Examples::    
        sage: weyl_lengths(["A",2])
        [1,-1,-1,1,1,-1]
        sage: weyl_lengths(["A",3])[:10]
        [1, -1, -1, -1, 1, 1, 1, 1, 1, -1]
    
    COMMENTS:
    Weyl vectors that are computed are stored in a dictionary _weyl_groups 
    for easy and quick access.
    """
    
    if str(type_rank) not in _weyl_lengths.keys():
        w_gr = weyl_group(type_rank)

        _weyl_lengths[str(type_rank)] = [det(g) for g in w_gr]
    return _weyl_lengths[str(type_rank)]

def lattice_norm(type_rank, v1, v2=None, basis=None,):
    """
    Compute the inner produt on the root or weight lattice  of a given
    Lie algebra between vectors v1 and v2.

    Input:

    -    ``type_rank`` - [str,int]; Lie group identifier
    -    ``v1`` -- vector; An lattice vector
    -    ``v2`` -- vector; An lattice vector
    -    ``basis`` -- string; basis of vectors, either root or weight

    Example:
        sage: vec1, vec2 = vector([1,2]), vector([2,3])
        sage: lattice_norm(["A",2],vec1, basis="weight")
            14/3
        sage: lattice_norm(["A",2],vec1, basis="root")
            6
        sage: vec3, vec4 = vector([1,2,3]), vector([4,5,6])
        sage: lattice_norm(["A",3],vec3, vec4, basis="root")
            24
    """

    if basis == None:
        warnings.warn("No basis is specified, weight is assumed.")
        basis = "weight"

    if v2 == None:
        v2 = v1

    assert len(v1) == len(v2), "Vectors must have same dimension"
    assert len(v1) == type_rank[1], "Dimensions do not match"

    mat = cartan_matrix(type_rank)
    if basis == "weight":
        mat = mat.inverse()

    return vector(v1)*mat*vector(v2)


_known_multiplicities = dict()
def freudenthal_multiplicity(type_rank, L, m):
    """
        Compute multiplicity of weight m within rep of highest weight L. 
        Uses Freudenthal\'s recursive formula.

        Input:
        -    ``type_rank`` - [str,int]; Lie group identifier
        -   ``Lambda`` -- vector; Highest weight vector
        -   ``m`` -- vector; Weight vector

        Example::
            sage: freudenthal_multiplicity(["A",2],vector([3,3]),vector([-1,2]))
            3
            sage: freudenthal_multiplicity(["A",3],vector([2,0,2]),vector([0,0,0]))
            6
    """

    cart = cartan_matrix(type_rank)
    cart_i = cart.inverse()
    rho = cart*weyl_vector(type_rank)

    if str([type_rank,L,m]) in _known_multiplicities.keys():
        return _known_multiplicities[str([type_rank,L,m])]
    mult = 0
    if np.logical_or.reduce(np.less(cart_i*(L-m), 0)):
        # m is a higher weight that Lambda
        mult = 0
    elif L == m:
        # m is Lambda
        mult = 1
    else:
        # m is a lower weight than Lambda
        num = 0
        p_roots = WeylGroup(type_rank).canonical_representation().positive_roots()
        for pr in p_roots:
            k = 1
            v = m+k*cart*pr
            while all(c >= 0 for c in cart_i*(L-v)):
                num += 2*freudenthal_multiplicity(type_rank,L,v) * \
                    lattice_norm(type_rank,v,cart*pr,basis="weight")
                k += 1
                v = m+k*cart*pr
        
        den = lattice_norm(type_rank,L+rho, basis="weight") - \
            lattice_norm(type_rank,m+rho, basis="weight")

        if den == 0:
            mult = 0
        else:
            mult = num / den
    _known_multiplicities[str([type_rank,L,m])] = mult
    return mult


def q_series(fprefexp, expMax, dim, qvar=None, rejection=3):
    """
        Compute the q series with prefactors and exponents computed by fprefexp.

        INPUT:
        -   ``fprefexp`` -- function; function which computes prefactors and exponents. Must take as input a dim-dimensional lattice vector only, and return prefactors and exponents as a tuple: (pref,exp), where pref and exp are lists.
        -   ``expMax`` -- Integer; maximum power of expansion
        -   ``qvar`` -- sage.symbolic.expression.Expression (default=var('q')); expansion variable
        -   ``rejection`` -- Integer; rejection parameter. After no acceptable terms are found in centered taxi-cab circles of increasing radius a number of times specified by the rejection parameter the function concludes.

        EXAMPLES::
            sage: fprefexp = lambda n: ([1/2*abs(n[0])],[n[0]^2]); q_series(fprefexp,120,1)
            10*q^100 + 9*q^81 + 8*q^64 + 7*q^49 + 6*q^36 + 5*q^25 + 4*q^16 + 3*q^9 + 2*q^4 + q

            sage: fprefexp = lambda n: ([1/2*abs(n[0])],[abs(n[0])]); q_series(fprefexp,10,1,qvar=var("z"))
            9*z^9 + 8*z^8 + 7*z^7 + 6*z^6 + 5*z^5 + 4*z^4 + 3*z^3 + 2*z^2 + z

            sage: fprefexp = lambda n: ([1/2*abs(n[0]+n[1])],[n[0]^2+n[0]*n[1]+n[1]^2]); q_series(fprefexp,20,2,rejection=5)
            20*q^19 + 8*q^16 + 16*q^13 + 8*q^12 + 6*q^9 + 12*q^7 + 4*q^4 + 4*q^3 + 2*q
    """
    amax = 0
    rejected = 0
    q_series = 0
    allNs = set()
    if qvar == None:
        qvar = var("q")

    while rejected < rejection:
        newNs = set(itertools.product(range(-amax, amax+1), repeat=dim))-allNs
        allNs = allNs.union(newNs)
        pref = list()
        exp = list()
        for n in newNs:
            temp = fprefexp(vector(n))
            pref += temp[0]
            exp += temp[1]

        new_terms_found = False
        for e, p, in zip(exp, pref):
            if abs(e) < expMax:
                new_terms_found = True
                rejected = 0
                q_series += p*qvar ** e
        # If there are no terms to keep (goodTerms is a list of False) increase rejected counter
        if not new_terms_found:
            rejected += 1
        amax += 1
    return q_series


def weyl_cycle(type_rank,v, f, z=None, basis=None):
    """
        Compute the Weyl Cycled z polynomial associated to v:

        MATH::
            \\sum_{w \\in W} (-1)^{f l(w)} \\exp^{\\langle  \\vec{\\xi}, w(v) \\rangle} =   0

        where z_i are defined in equations (3.20) and (3.21) of [1].

        INPUT:
        -    ``type_rank`` - [str,int]; Lie group identifier
        -   ``v`` -- vector; lattice vector
        -   ``f`` -- Integer; Weyl length factor
        -   ``z`` -- variable (Optional); symbolic expressions of z_i. If none are given then z_i is chosen as default.
        -   ``basis`` -- string; basis in which v is given.

        EXAMPLES::
            sage: weyl_cycle(["A",2],vector([1,2]),3,basis = "weight")
            z0*z1^2 - z1^3/z0 - z0^3/z1^2 + z0^2/z1^3 + z1/z0^3 - 1/(z0^2*z1)
            sage: weyl_cycle(["A",3],vector([1,2,1]),3,basis = "weight")
            z0*z1^2*z2 - z0^3*z2^3/z1^2 - z0*z1^3/z2 - z1^3*z2/z0 + z0^4*z2^2/z1^3 
            +z0^2*z2^4/z1^3 + z1^4/(z0*z2) - z0^3*z2^3/z1^4 + z0^3*z1/z2^3 
            -z0^4/(z1*z2^2) + z1*z2^3/z0^3 - z2^4/(z0^2*z1) - z0^2*z1/z2^4 
            +z0^3/(z1*z2^3) - z1*z2^2/z0^4 + z2^3/(z0^3*z1) - z1^4/(z0^3*z2^3) 
            +z0*z2/z1^4 + z1^3/(z0^2*z2^4) + z1^3/(z0^4*z2^2) - z0/(z1^3*z2) 
            -z2/(z0*z1^3) - z1^2/(z0^3*z2^3) + 1/(z0*z1^2*z2)   
    """

    if basis == None:
        warnings.warn("No basis is specified, weight is assumed.")
        basis = "weight"

    assert basis == "root" or basis == "weight", "basis must be root or weight"

    rk = type_rank[1]    
    if z == None:
        varstr = ""
        for i in range(rk):
            varstr += f", z{i}"
        z = var(varstr)

    if basis == "root":
        v = cartan_matrix(type_rank)*v
        basis = "weight"

    WG = [g.transpose() for g in weyl_group(type_rank)] # get to weight basis
    WL = weyl_lengths(type_rank)

    v = vector(v)
    WGv = list()
    for g, l in zip(WG, WL):
        WGv.append([g*v, l])

    cycle = list()
    for gv in WGv:
        cycle.append(gv[1]**(f) *
                     product([x**y for x, y in zip(z, gv[0])]))
    return sum(cycle)


def const_term(expr):
    """
    Extract the constant term of a polyonmial.

    INPUT:
    -   ``expr`` -- polynomial

    EXAMPLES::
        sage: x = var("x");const_term(x^2+x+3)
        3
    """
    q = var("q")
    zeroTerm = expr
    for v in expr.variables():
        if v == q:
            continue
        for coeff, exp in zeroTerm.series(v,3).coefficients(v):
            if exp == 0:
                zeroTerm = coeff
                break
    return zeroTerm


def weyl_exp(N):
    """
    Expand the inverse of the A_2 Weyl determinant in powers of z_i, for large z_i.

    INPUT:
    -   ``N`` -- 2 dimensional vector; expansion order

    EXAMPLES::
        sage: weyl_exp(vector([2,3]))
        (z0/z1^2 + z0^2/z1^4 + 1)*(z1/z0^2 + z1^2/z0^4 + z1^3/z0^6 + 1)*(1/(z0*z1) + 1/(z0^2*z1^2) + 1)/(z0*z1)

    """

    z0, z1 = var("z0,z1")
    return 1/(z0*z1) * sum([(z0/z1 ^ 2) ^ n for n in range(N[0]+1)]) * sum([(z1/z0 ^ 2) ^ n for n in range(N[1]+1)])*sum([(1/(z0*z1)) ^ n for n in range(min(N)+1)])


def triplet_character(type_rank, lmbd, mu, m, f, expMax,  basis="weight", qvar=var("q")):
    """
    Compute the triplet character with specified parameters, up to the inverse Dedekind eta function to the
    power of the rank of the lattice. Argument descriptions refer to equation (3.15) of [1].

    INPUTS:
    -    ``type_rank`` - [str,int]; Lie group identifier
    -   ``lmbd`` -- Vector; lambda parameter in equation (3.15)
    -   ``mu`` -- Vector; mu parameter in equatoin (3.13)
    -   ``m`` -- Integer; m parameter in equatoin (3.13)
    -   ``f`` -- Integer; number of fibers of Seifert manifold
    -   ``expMax`` -- Integer; Maximum exponent in q series expansion
    -   ``basis`` -- String; basis in which wh and b are given
    -   ``qvar`` -- Variable (default=None); variable in which to expand. If None qvar = var("q")

    EXAMPLES::
            sage: lmbd,mu,m,f,expMax = vector([0,0]),1/sqrt(30)*vector([3,3]),30,2,20
            sage: triplet_character(["A",2],lmbd,mu,m,f,expMax)
            6*q^(2/15)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1))
    """

    rk = type_rank[1]
    cart_i = cartan_matrix(type_rank).inverse()
    if basis == "weight":
        mu = cart_i*mu
        lmbd = cart_i*lmbd
        basis = "root"

    rho = weyl_vector(type_rank) # Basis is root

    def Delta(v): 
        return weyl_cycle(type_rank, v, f, basis=basis)

    def fprefexp(lt):
        expl = [1/(2*m)*lattice_norm(type_rank, m*(lmbd+lt) +
                                     sqrt(m)*mu+(m-1)*rho, basis=basis)]
        prefl = [Delta(lt+lmbd+rho)/Delta(rho)]
        return prefl, expl

    return q_series(fprefexp, expMax, rk, qvar=qvar)


def singlet_character(type_rank, lmbdt, lmbd, mu, m, f, expMax, basis="root", qvar=var("q")):
    """
    Compute the singlet character with specified parameters, up to the inverse Dedekind eta function to the
    power of the rank of the lattice. Argument descriptions refer to equation (3.20) of [1].

    INPUT:
    -   ``type_rank`` - [str,int]; Lie group identifier
    -   ``lmbd`` -- Vector; lambda-tilde parameter in equation (3.20)
    -   ``lmbd`` -- Vector; lambda parameter in equation (3.20)
    -   ``mu`` -- Vector; mu parameter in equatoin (3.20)
    -   ``m`` -- Integer; m parameter in equatoin (3.20)
    -   ``f`` -- Integer; number of fibers of Seifert manifold
    -   ``expMax`` -- Integer; Maximum exponent in q series expansion
    -   ``basis`` -- String; basis in which wh and b are given
    -   ``qvar`` -- Variable (default=None); variable in which to expand. If None qvar = var("q")

    EXAMPLES::
        sage: lmbd,lmbdt,mu,expMax = vector([0,0]),vector([0,0]),1/sqrt(S.m)*vector([3,3]),100
        ....: singlet_character(lmbdt,lmbd,mu,m,f,expMax)
        -4*q^(1352/15) - 4*q^(1262/15) - q^(512/15) - 2*q^(482/15) - 2*q^(422/15) - q^(392/15)
    """

    rk = type_rank[1]
    cart_i = cartan_matrix(type_rank).inverse()
    if basis == "weight":
        mu = cart_i*mu
        lmbd = cart_i*lmbd
        basis = "root"

    rho = weyl_vector(type_rank) # Basis is root

    varstr = ""
    for i in range(rk):
        varstr += f", z{i}"
    z = var(varstr)

    def Delta(v): 
        return weyl_cycle(type_rank, v, f, basis=basis)

    def fprefexp(lt):
        expl = [1/(2*m)*lattice_norm(type_rank, m*(lmbd+lt) +
                                     sqrt(m)*mu+(m-1)*rho, basis=basis)]
        prefl = [const_term(Delta(lt+lmbd+rho)/Delta(rho)*product([zz**l for zz, l in zip(z,lmbdt)]))]
        return prefl, expl

    return q_series(fprefexp, expMax, rk, qvar=qvar)



def triplet_character_p_pprime(p, pp, s, r, expMax, qvar=var("q"), z=var("z")):
    """
    Compute the p,p' triplet character with specified parameters, up to the inverse Dedekind eta function.
    Argument descriptions refer to equation (3.45) of [1].

    INPUT:
    -   ``p`` -- Vector; lambda parameter in equation (3.20)
    -   ``pp`` -- Vector; mu parameter in equatoin (3.20)
    -   ``s`` -- Integer; m parameter in equatoin (3.20)
    -   ``r`` -- Integer; number of fibers of Seifert manifold
    -   ``expMax`` -- Integer; Maximum exponent in q series expansion
    -   ``basis`` -- String; basis in which wh and b are given
    -   ``qvar`` -- Variable (default=None); variable in which to expand.

    EXAMPLES::
        sage: p, pp, r, s, expMax = 2, 105, 29, 1,200
        ....: triplet_character_p_pprime(p,pp,r,s,expMax)
        -(z^2 + 1/z^2 - 2)*q^(139129/840)/(z - 1/z)^2 + (z^2 + 1/z^2 - 2)*q^(66049/840)/(z - 1/z)^2

    """

    def fprefexp(n):
        k = n[0]
        prefl = [x*(z ^ (2*k)-2+z ^ (-2*k))*(z-z ^ (-1)) ^ (-2)
                 for x in [1, -1]]
        expl = [(2*p*pp*k+p*s+pp*r) ^ 2/(4*p*pp),
                (2*p*pp*k+p*s-pp*r) ^ 2/(4*p*pp)]
        return prefl, expl

    return q_series(fprefexp, expMax, 1, qvar=qvar)


def _continued_fraction(x, iterMax=1000):
    """
    Computes the continued fraction of x with parameter a up to iterMax iterations.

    INPUT:
    -   ``x`` --  Rational; fraction to expand
    -   ``iterMax`` - Integer (default = 1000); iteration maximum, default = 1000

    EXAMPLES::
        sage: _continued_fraction(3/5)
        [1,3,2]

    Some code taken from
    https://share.cocalc.com/share/d1efa37e-be6a-40f6-80c7-8a34201d7c4e/PlumbGraph.sagews?viewer=share
    on July 4th 2021

    """
    assert x in QQ, "x must be a rational number"
    assert iterMax in NN, "iterMax must be a positive integer"

    a = -1
    n = 0
    r = list()
    while n < iterMax:
        r.append(ceil(x))
        if x == ceil(x):
            break
        else:
            x = 1/(ceil(x)+a*x)
            n += 1
    if n < iterMax:
        return r
    else:
        return r


class Seifert():
    #
    def __init__(self, SeifertData, qvar=var('q')):
        """
        Create a Seifert manifold.

        Implements the "Seifert" constructor. Currently working for Seifert manifolds with three and four exceptional fibers.

        INPUT:
        -   ``SeifertData`` -- list; List containing the Seifert manifold data, as [b,q_1,p_1,...,p_n,q_n]
        -   ``qvar`` -- q variable for q series

        EXAMPLES::
            sage: S = Seifert([-1, 1, 2, 1, 3, 1, 5]);S
            Seifert manifold with 3 exceptional fibers.
            Seifert data:
            [-1, 1, 2, 1, 3, 1, 5]
            Plumbing Matrix:
            [-1  1  1  1]
            [ 1 -2  0  0]
            [ 1  0 -3  0]
            [ 1  0  0 -5]
            D: 1, m: 30, det(M): -1

            sage: S = Seifert([-1, 1, 2, 1, 3, 1, 5, 1, 7]);S
            Seifert manifold with 4 exceptional fibers.
            Seifert data:
            [-1, 1, 2, 1, 3, 1, 5, 1, 7]
            Plumbing Matrix:
            [-1  1  1  1  1]
            [ 1 -2  0  0  0]
            [ 1  0 -3  0  0]
            [ 1  0  0 -5  0]
            [ 1  0  0  0 -7]
            D: 37, m: 210, det(M): 37
        """
        self.SeifertData = SeifertData
        self.b = SeifertData[0]
        self.q = [SeifertData[x] for x in range(1, len(SeifertData), 2)]
        # p is the order of singular fibers
        self.p = [SeifertData[x] for x in range(2, len(SeifertData), 2)]
        self.f = len(self.q)
        assert self.f == 3 or self.f == 4, "pySeifert only currently supports three or four fibers"
        self.M = self._plumbing_matrix()
        self.L = self.M.ncols()
        self.Mdet = self.M.det()
        self.Minv = self.M.inverse()
        self.d = 1/GCD(self.Minv[0])
        self.m = abs(self.Minv[0, 0]*self.d)
        self.deg = self._degrees()  # returns 2-deg(v) of plumbing matrix
        self._legs()
        self.A = vector([x if y == 1 else 0 for x,
                        y in zip(self.Minv[0], self.deg)])
        self.qvar = qvar
        self.Coker = self._coker()

    def __repr__(self):
        """
        INPUT:
        -   ``self`` -- Seifert; Seifert namifold

        EXAMPLE::
            sage: S = Seifert([-1, 1, 2, 1, 3, 1, 5, 1, 7]);S
            Seifert manifold with 4 exceptional fibers.
            Seifert data:
            [-1, 1, 2, 1, 3, 1, 5, 1, 7]
            Plumbing Matrix:
            [-1  1  1  1  1]
            [ 1 -2  0  0  0]
            [ 1  0 -3  0  0]
            [ 1  0  0 -5  0]
            [ 1  0  0  0 -7]
            D: 37, m: 210, det(M): 37
        """
        return 'Seifert manifold with {} exceptional fibers.\nSeifert data:\n{}\nPlumbing Matrix:\n{}\nD: {}, m: {}, det(M): {}'.format(self.f, self.SeifertData, self.M, self.d, self.m, self.Mdet)

    def _latex_(self):
        """
        Print latex name
        """
        latex_name = f"M\\left({self.b};"
        for i in range(self.f):
            latex_name += "\\frac{"+str(self.q[i])+"}{"+str(self.p[i])+"}, "
        latex_name = latex_name[:-2] + "\\right)"
        return latex_name

    def _plumbing_matrix(self):
        r"""
        Compute the plumbing matrix of self.

        Some code taken from
        https://share.cocalc.com/share/d1efa37e-be6a-40f6-80c7-8a34201d7c4e/PlumbGraph.sagews?viewer=share
        on July 4th 2021
        """
        l = [len(_continued_fraction(self.SeifertData[2*i]/self.SeifertData[2*i-1]))
             for i in range(1, self.f+1)]
        M = matrix(1+sum(l))
        M[0, 0] = self.b
        for j in range(len(l)):
            for k in range(l[j]):
                if k == 0:
                    M[0, 1+sum(l[:j])] = 1
                    M[1+sum(l[:j]), 0] = 1
                    M[1+k+sum(l[:j]), 1+k+sum(l[:j])] = (-1)*_continued_fraction(
                        self.SeifertData[2*j+2]/self.SeifertData[2*j+1])[k]
                else:
                    M[1+k+sum(l[:j]), k+sum(l[:j])] = 1
                    M[k+sum(l[:j]), 1+k+sum(l[:j])] = 1
                    M[1+k+sum(l[:j]), 1+k+sum(l[:j])] = (-1)*_continued_fraction(
                        self.SeifertData[2*j+2]/self.SeifertData[2*j+1])[k]
        return M

    def _degrees(self):
        """
        Compute the degrees of the vertices in the order that they appear in the plumbing matrix of self. Returns 2-deg(v)
        """
        deg = list()
        for i, row in enumerate(list(self.M)):
            deg.append(2-(sum(row)-row[i]))
        return deg

    def _coker(self):
        """
        Compute the cokernel of the plumbing matrix (of self).
        """

        if self.Mdet**2 == 1:
            return [[0]*self.L]
        Coker = []
        for v in range(self.L):
            vec = vector([0]*self.L)
            for i in range(abs(self.Mdet)):
                vec[v] = i+1
                new = [x-floor(x) for x in self.Minv*vec]
                if new not in Coker:
                    Coker.append(new)
        return Coker

    def _legs(self):
        """
        Compute the lengths of the legs of the plumbing graph.
        """
        llen = 1
        self.legs = list()
        for d in reversed(self.deg[1:-1]):
            if d == 1:
                self.legs.append(llen)
                llen = 0
            llen += 1
        self.legs.append(llen)
        self.legs.reverse()

    def _zhat_prefactor(self,type_rank, qvar=None):
        """
        Compute the prefactor for the Zhat invariant of A_rk.
        """
        if qvar == None:
            qvar = self.qvar
        pos_eigen = sum(np.greater(self.M.eigenvalues(), 0))
        Phi = len(WeylGroup(type_rank).canonical_representation().positive_roots())
        sig = 2*pos_eigen - (self.L)
        norm_rho2 = lattice_norm(type_rank,weyl_vector(type_rank),basis="root")
        return (-1) ^ (Phi*pos_eigen)*q^(3*sig-self.M.trace()/2*norm_rho2)

    def delta(self, type_rank):
        """
        Compute the q-exponent delta as in equation (4.12) of [1]. Works for rk=1,2.

        INPUT:
        -   ``type_rank`` - [str,int]; Lie group identifier

        EXAMPLES::
            sage: S = Seifert([-1,1,2,1,3,1,5]);S.delta(["A",2])
            31/30
            sage: S.delta(["D",6])
            341/12

        """
        return sum([lattice_norm(type_rank,cartan_matrix(type_rank)*weyl_vector(type_rank), basis="weight")*1/2*(self.Minv[0, i] ^ 2/self.Minv[0, 0]-self.Minv[i, i]) for i, d in enumerate(self.deg) if d == 1])

    def boundary_conditions(self, type_rank, basis=None):
        """
        Compute representatives for the set of boundary conditions for the Seifert manifold with respect to A_rk group, as in equation (4.4) of [1]. Works for rk=1,2.

        INPUT:
        -   ``type_rank`` - [str,int]; Lie group identifier
        -   ``basis`` -- str; basis in which the bs are to be outputted.

        EXAMPLES::
            sage: S = Seifert([-1,1,2,1,3,1,5])
            ....: S.boundary_conditions(["A",2], basis = "weight")
            [[(-1, -1), (1, 1), (1, 1), (1, 1)]]

            sage: S = Seifert([-1,1,2,1,2,1,2])
            ....: S.boundary_conditions(["A",2], basis = "root")
            [[(-1, -1), (1, 1), (1, 1), (1, 1)],
             [(0, -1), (0, 1), (0, 1), (1, 1)],
             [(0, -1), (0, 1), (1, 1), (0, 1)],
             [(0, -1), (1, 1), (0, 1), (0, 1)]]

            sage: S = Seifert([-1,1,2,1,2,1,2])
            ....: S.boundary_conditions(["D",4],basis="root")
            [[(-3, -5, -3, -3), (3, 5, 3, 3), (3, 5, 3, 3), (3, 5, 3, 3)],
             [(-2, -5, -3, -3), (2, 5, 3, 3), (2, 5, 3, 3), (3, 5, 3, 3)],
             [(-2, -5, -3, -3), (2, 5, 3, 3), (3, 5, 3, 3), (2, 5, 3, 3)],
             [(-2, -5, -3, -3), (3, 5, 3, 3), (2, 5, 3, 3), (2, 5, 3, 3)]]

        """

        if basis == None:
            warnings.warn("No basis is specified, weight is assumed")
            basis = "weight"
        
        # If Brieskorn sphere, just return the b0
        rho = weyl_vector(type_rank)
        b0 = [d*rho for d in self.deg]
        if self.Mdet**2 == 1:
            if basis == "weight":
                cart = cartan_matrix(type_rank)
                b0 = [d*cart*rho for d in self.deg]
            return [b0]

        # Tensor (Coker otimes Lambda)
        rk = type_rank[1]
        e = identity_matrix(rk)

        boundary_conditions = list()
        for ei, v in itertools.product(e, self.Coker):
            boundary_conditions.append([x*ei for x in self.M*vector(v)])
        # Calculate the orbit of each connection w/r. to the Weyl group.
        # If an orbit element is already present in boundary_conditions, then remove connection.
        toRemove = set()
        for i, b in enumerate(boundary_conditions):
            for g in weyl_group(type_rank):
                gb = [g*vector(x) for x in b]
                remove = False
                for bb in boundary_conditions[:i]:
                    if vector(self.Minv*(matrix(gb)-matrix(bb))) in ZZ ^ (self.L*rk):
                        remove = True
                        toRemove.add(i)
                        break
                if remove:
                    break
        
        for i in reversed(sorted(toRemove)):
            del boundary_conditions[i]

        boundary_conditions = sorted(
            [list(matrix(b)+matrix(b0)) for b in boundary_conditions])

        if basis == "weight":
            C = cartan_matrix(type_rank)
            for i in range(len(boundary_conditions)):
                b = [C*v for v in boundary_conditions[i]]
                boundary_conditions[i] = b
                
        return boundary_conditions

    def S_set(self, type_rank, whr, b, basis="root"):
        """
        Compute the set \\kappa_{\\hat{w};\\vec{\\underline{b}}}, as in equation (4.28) of [1].

        INPUTS:
        -   ``type_rank`` - [str,int]; Lie group identifier
        -   ``whr`` -- list of length self.L of vectors of rank rk; list of lattice vectors. whr[i] should be the zero vector if self.deg[i] is different than 1, w_i*rho otherwise
        -   ``b`` --  list of length self.L of vectors of rank rk; boundary condition
        -   ``basis`` -- basis in which whr and b are given

        EXAMPLES::
        sage: S = Seifert([0,1,3,1,2,1,6])
        ....: b = S.boundary_conditions(["A",2], basis = "root")[1]
        ....: rho = vector([1,1])
        ....: whr = [identity_matrix(2)*rho if d == 1 else matrix(2)*rho for d in S.deg]
        ....: S.S_set(["A",2],whr,b)
        [(1, 5)]

        """

        if basis == "weight":
            cart_i = cartan_matrix(type_rank).inverse()
            whr = [cart_i*vector(v) for v in whr]
            b = [cart_i*vector(v) for v in b]
        
        rk = type_rank[1]
        rho = weyl_vector(type_rank)
        whr = matrix(whr) + matrix([- (type_rank[1] % 2) * rho]+ [[0]*rk]*(len(whr)-1))
        lam = - matrix(whr) - matrix(b)
        k_list = list()
        MS = MatrixSpace(ZZ, lam.nrows(), lam.ncols())
        eps = rk % 2
        for k in itertools.product(range((1+eps)*self.d),repeat=rk):
            kappa = matrix([vector(k)/(1+eps)+eps*rho]+[[0]*rk]*(lam.nrows()-1))
            if self.Minv*(kappa+lam) in MS:
                k_list += [vector(k)]
        if basis == "weight":
            cart = cartan_matrix(type_rank)
            k_list = [cart*v for v in k_list]
        return k_list

    def s_values(self, type_rank, b, basis="weight", nu=None, wilVert=0):
        """
        Compute the set of \\vec s values, and their respective \\kappa_{\\hat w; \\vec{\\underline b}} and Weyl length, as in equations (4.28) (\\kappa) and (4.36) (\\vec s) of [1].

        INPUT:
        -   ``type_rank`` - [str,int]; Lie group identifier
        -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
        -   ``basis`` -- String; basis in which b is given.
        -   ``WGsimb`` -- Bool; if True, print a hat-w in symbolic form for every s value
        -   ``nu`` -- Vector; Highest weight of Wilson line operator to be attached at an end node. If None, no Wilson operator is attached.
        -   ``wilVert`` -- Integer; leg at which to attach the Wilson operator

        Example:

        
        sage: S = Seifert([0,1,3,1,2,1,6])
        ....: b = S.boundary_conditions(["A",2],basis = "root")[0]
        ....: S.s_values(["A",2],b,"weight")
        [[-1, (-6, -6), (5, 5)],
         [1, (0, 0), (4, 4)],
         [1, (6, -12), (5, 5)],
         [-1, (0, 0), (6, 3)],
         [1, (-12, 6), (5, 5)],
         [-1, (0, 0), (3, 6)],
         [1, (0, 0), (10, -5)],
         [-1, (-6, 12), (5, 5)],
         [1, (0, 0), (-5, 10)],
         [-1, (12, -6), (5, 5)],
         [-1, (0, 0), (0, 0)],
         [1, (6, 6), (5, 5)]]

        sage: S = Seifert([0,1,3,1,2,1,6])
        ....: b = S.boundary_conditions(["A",1],basis = "weight")[0]
        ....: S.s_values(["A",1],b,"weight")
        [[-1, (-6), (10)], [1, (6), (10)]]

        """

        rk = type_rank[1]

        if nu == None:
            nu = vector([0]*rk)

        WG = weyl_group(type_rank)
        if basis == "weight":
            WG = [g.transpose() for g in WG]
        WL = weyl_lengths(type_rank)
        it = iter([[1]*len(WG) if d == wilVert else [0]*len(WG)
                  for d in range(self.f)])
        
        rho = weyl_vector(type_rank)
        if basis == "weight":
            rho = cartan_matrix(type_rank).inverse()*rho
        
        WGr = [[g*(rho+i*nu) for g, i in zip(WG, next(it))] if d ==
               1 else [vector([0]*rk)] for d in self.deg]

        whrl = itertools.product(*WGr)
        whlenl = itertools.product(reversed(WL), repeat=self.f)
        alls_values = list()
        for whr, whlen in zip(whrl, whlenl):
            S_set = self.S_set(type_rank, whr, b, basis=basis)
            if S_set != []:
                new = [product(whlen),
                       (-self.d*vector(self.A)*matrix(whr)), S_set[0]]
                alls_values.append(new)
        return alls_values

    def chi_tilde(self, type_rank, wh, b, expMax, basis="weight", qvar=None):
        """
            Compute the chi_tilde q-series like in equation (4.34) of [1].

            INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- q variable for q series

            EXAMPLES::
                sage: S = Seifert([0,1,3,1,2,1,6]);
                ....: wh = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg];
                ....: b = S.boundary_conditions(2,basis = "weight")[0];
                ....: expMax = 20;
                ....: S.chi_tilde(["A",2],wh, b, expMax, basis = "weight")
                (z0^5*z1^5 - z0^10/z1^5 - z1^10/z0^5 + z0^5/z1^10 + z1^5/z0^10 - 1/(z0^5*z1^5))*q^17/(z0*z1 - z0^2/z1 - z1^2/z0 + z0/z1^2 + z1/z0^2 - 1/(z0*z1)) - q^5

                sage: S = Seifert([0,1,3,1,2,1,6]);
                ....:  wh = [identity_matrix(1) if d == 1 else matrix(1) for d in S.deg];
                ....:  b = S.boundary_conditions(["A",1],basis = "weight")[0];
                ....:  expMax = 20;
                ....:  S.chi_tilde(["A",1],wh, b, expMax, basis = "weight")
                -q^(5/4)

                sage: S = Seifert([0,1,3,1,2,1,6]);
                ....: wh = [identity_matrix(3) if d == 1 else matrix(3) for d in S.deg];
                ....: b = S.boundary_conditions(["A",3],basis = "weight")[0];
                ....: expMax = 20;
                ....: S.chi_tilde(["A",3],wh, b, expMax, basis = "weight")
                -(z0^4*z1*z2^4 - z0^5*z2^5/z1 - z0^4*z1^5/z2^4 + z0^9*z2/z1^5 - z1^5*z2^4/z0^4 + z0*z2^9/z1^5 + z0^5*z1^4/z2^5 - z0^9/(z1^4*z2) + z1^4*z2^5/z0^5 - z2^9/(z0*z1^4) + z1^9/(z0^4*z2^4) - z0^5*z2^5/z1^9 - z1^9/(z0^5*z2^5) + z0^4*z2^4/z1^9 - z0*z1^4/z2^9 + z0^5/(z1^4*z2^5) - z1^4*z2/z0^9 + z2^5/(z0^5*z1^4) + z1^5/(z0*z2^9) - z0^4/(z1^5*z2^4) + z1^5/(z0^9*z2) - z2^4/(z0^4*z1^5) - z1/(z0^5*z2^5) + 1/(z0^4*z1*z2^4))*q^(25/2)/(z0*z1*z2 - z0^2*z2^2/z1 - z0*z1^2/z2 + z0^3*z2/z1^2 - z1^2*z2/z0 + z0*z2^3/z1^2 + z0^2*z1/z2^2 - z0^3/(z1*z2) + z1^3/(z0*z2) - z0^2*z2^2/z1^3 + z1*z2^2/z0^2 - z2^3/(z0*z1) - z0*z1/z2^3 + z0^2/(z1*z2^2) - z1^3/(z0^2*z2^2) + z0*z2/z1^3 - z1*z2/z0^3 + z2^2/(z0^2*z1) + z1^2/(z0*z2^3) - z0/(z1^2*z2) + z1^2/(z0^3*z2) - z2/(z0*z1^2) - z1/(z0^2*z2^2) + 1/(z0*z1*z2)) + q^(25/2)
        """
        rk = type_rank[1]
        if qvar == None:
            qvar = self.qvar

        cart_i = cartan_matrix(type_rank).inverse()
        if basis == "weight":
            wh = [w.transpose() for w in wh]
            b = [cart_i*v for v in b]
            basis = "root"
        rho = weyl_vector(type_rank)

        whr = [w*rho for w in wh]
        Aw = -1/self.Minv[0, 0]*self.A*matrix(whr)
        kappa = self.S_set(type_rank, whr, b, basis=basis)

        if kappa == []:
            return 0
        else:
            kappa = kappa[0]

        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)

        D_r = Delta(rho)
        eps = rk % 2
        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [self.delta(type_rank)+self.m/(2*self.d) *
                    lattice_norm(type_rank, v+Aw, basis=basis)]
            prefl = [Delta(v)/D_r^(self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)

    def chi_tilde_wilson_end(self, type_rank, wh, b, expMax, nu, leg, basis="weight", qvar=None):
        """
        Compute the tilde_chi q-series with Wilson operator insertion at an end node, as described in section 4.3 of [1].

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``nu`` -- Vector; highest weight of representation
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- Variable; q variable for q series

        EXAMPLES::
            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: wh = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg];
            ....: b = S.boundary_conditions(["A",2],basis = "weight")[0];
            ....: expMax = 100;
            ....: nu, leg = vector([3,3]), 0;
            ....: S.chi_tilde_wilson_end(["A",2],wh, b, expMax, nu, leg, basis = "weight", qvar=None)
            (z0^4*z1^4 + z0^8/z1^4 + z1^8/z0^4 + z0^4/z1^8 + z1^4/z0^8 + 1/(z0^4*z1^4))*q^(175/9)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1)) + (z0^2*z1^2 + z0^4/z1^2 + z1^4/z0^2 + z0^2/z1^4 + z1^2/z0^4 + 1/(z0^2*z1^2))*q^(103/9)/(z0*z1 + z0^2/z1 + z1^2/z0 + z0/z1^2 + z1/z0^2 + 1/(z0*z1))

            sage: S = Seifert([-1,1,3,1,4,1,5]);
            ....: wh = [identity_matrix(3) if d == 1 else matrix(3) for d in S.deg];
            ....: b = S.boundary_conditions(["A",3],basis = "weight")[0];
            ....: expMax = 200;
            ....: nu, leg = vector([1,2,1]), 0;
            ....: S.chi_tilde_wilson_end(["A",3],wh, b, expMax, nu, leg, basis = "weight", qvar=None)
            (z0*z1^5*z2^5 - z0^6*z2^10/z1^5 - z1^6*z2^5/z0 + z0^5*z2^11/z1^6 - z0*z1^10/z2^5 + z0^11*z2^5/z1^10 + z1^11/(z0*z2^5) - z0^10*z2^6/z1^11 + z1*z2^10/z0^6 - z2^11/(z0^5*z1) + z0^6*z1^5/z2^10 - z0^11/(z1^5*z2^5) - z0^5*z1^5/z2^11 + z0^10/(z1^5*z2^6) - z1^11/(z0^6*z2^10) + z0^5*z2/z1^11 - z1*z2^5/z0^11 + z2^6/(z0^10*z1) + z1^10/(z0^5*z2^11) - z0^5/(z1^10*z2) + z1^6/(z0^11*z2^5) - z2/(z0^5*z1^6) - z1^5/(z0^10*z2^6) + 1/(z0^5*z1^5*z2))*q^(822927/4394)/(z0*z1*z2 - z0^2*z2^2/z1 - z0*z1^2/z2 + z0^3*z2/z1^2 - z1^2*z2/z0 + z0*z2^3/z1^2 + z0^2*z1/z2^2 - z0^3/(z1*z2) + z1^3/(z0*z2) - z0^2*z2^2/z1^3 + z1*z2^2/z0^2 - z2^3/(z0*z1) - z0*z1/z2^3 + z0^2/(z1*z2^2) - z1^3/(z0^2*z2^2) + z0*z2/z1^3 - z1*z2/z0^3 + z2^2/(z0^2*z1) + z1^2/(z0*z2^3) - z0/(z1^2*z2) + z1^2/(z0^3*z2) - z2/(z0*z1^2) - z1/(z0^2*z2^2) + 1/(z0*z1*z2))
        """

        rk = type_rank[1]
        if qvar == None:
            qvar = self.qvar

        if basis == "weight":
            wh = [w.transpose() for w in wh]
            cart_i = cartan_matrix(type_rank).inverse()
            nu = cart_i*nu
            b = [cart_i*v for v in b]
            basis = "root"
        rho = weyl_vector(type_rank)

        whr = [w*rho for w in wh]
        end_ind = sum([self.legs[i] for i in range(leg+1)])
        whr[end_ind] += wh[end_ind]*nu
        Aw = -1/self.m*self.A*matrix(whr)
        kappa = self.S_set(type_rank, whr, b, basis=basis)
        if kappa == []:
            return 0
        else:
            kappa = kappa[0]

        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)
        D_r = Delta(rho)

        eps = rk % 2
        d = self.delta(type_rank) + (lattice_norm(type_rank,rho+nu, basis=basis)- \
            lattice_norm(type_rank, rho, basis=basis)) / \
            2*(self.Minv[0, end_ind] ^ 2 /
               self.Minv[0, 0]-self.Minv[end_ind, end_ind])
               

        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [d+self.m/(2*self.d)*lattice_norm(type_rank, v+Aw, basis=basis)]
            prefl = [Delta(v)/D_r ^ (self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)

    def chi_tilde_wilson_mid(self, type_rank, wh, wp, b, expMax, sig, leg, step, basis="weight", qvar=None):
        """
        Compute the chi_tilde q-series with Wilson operator insertion at an intermediate node as in equation (4.81) of [1].

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``wp`` -- Matrix; w' Weyl group element
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``sig`` -- Vector; weight of representation
            -   ``leg`` -- Integer; leg at which to attach the Wilson operator
            -   ``nu`` -- Vector; highest weight of representation
            -   ``step`` -- Integer; step in leg at which to attach the Wilson operator
            -   ``basis`` -- String; basis in which b is given.
            -   ``qvar`` -- Variable (optional, default = self.qvar); q variable for q series

        EXAMPLES::
            sage: S = Seifert([-1,2,3,-1,2,-1,2]);
            ....: wh,wp = [identity_matrix(2) if d == 1 else matrix(2) for d in S.deg], identity_matrix(2)
            ....: b = S.boundary_conditions(["A",2],basis = "weight")[0];
            ....: expMax = 19;
            ....: sig, leg,step = vector([1,1]), 0, 1;
            ....: S.chi_tilde_wilson_mid(["A",2], wh, wp, b, expMax, sig, leg, step, basis = "weight", qvar=None)
            q^(67/64)
        """
        assert not (step == 0 or step ==
                    self.legs[leg]), "No center or endpoint"

        rk = type_rank[1]
        
        if qvar == None:
            qvar = self.qvar

        if basis == "weight":
            wh = [w.transpose() for w in wh]
            wp = wp.transpose()
            cart_i = cartan_matrix(type_rank).inverse()
            sig = cart_i*sig
            b = [cart_i*v for v in b]
            basis = "root"
        
        rho = weyl_vector(type_rank)

        whr = [w*rho for w in wh]
        wil_ind = sum([self.legs[i] for i in range(leg-1)])+step
        whr[wil_ind] += wp*sig

        Aw = -1/self.m*self.A * \
            matrix(whr)-self.Minv[0, wil_ind]/self.Minv[0, 0]*wp*sig

        kappa = self.S_set(type_rank, whr, b, basis=basis)
        if kappa == []:
            return 0
        else:
            kappa = kappa[0]

        def Delta(v): return weyl_cycle(type_rank, v, self.f, basis=basis)
        D_r = Delta(rho)

        eps = rk % 2
        d = self.delta(type_rank) + sum([lattice_norm(type_rank, wr, wp*sig, basis=basis)*(self.Minv[0, wil_ind]*self.Minv[0, i]/self.Minv[0, 0]-self.Minv[i, wil_ind])
                                  for i, wr in enumerate(whr)])-1/2*lattice_norm(type_rank, sig, basis=basis)*(self.Minv[0, wil_ind] ^ 2/self.Minv[0, 0]-self.Minv[wil_ind, wil_ind])

        def fprefexp(l):
            v = self.d*l+kappa+eps*rho
            expl = [d+self.m/(2*self.d)*lattice_norm(type_rank, v+Aw, basis=basis)]
            prefl = [Delta(v)/D_r ^ (self.f-2)]
            return prefl, expl

        return q_series(fprefexp, expMax, rk, qvar=qvar)

    def chi_prime_4f_sph(self, wh, expMax, basis=None, qvar=None):
        """
        Compute the hat Z integrand for four fibered sphereical and pseudospherical examples with pecified parameters. Argument descriptions refer to equation (4.53) of [1].

        INPUTS:
            -   ``wh`` -- list of length self.N of matrices; \\hat w vector. Must be of same length as the number of nodes, with matrices with zero entries at nodes with degree - 2 != 1
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``basis`` -- String (optional, default = weight); basis in which wh is given.
            -   ``qvar`` -- Variable (optional, default = self.qvar); q variable for q series
        EXAMPLES::
            sage: S = Seifert([-2, 1, 2, 2, 3, 2, 5, 3, 7]);
            ....: wh, expMax =  [identity_matrix(1) if d == 1 else matrix(1) for d in S.deg], 100;
            ....: S.chi_prime_4f_sph(wh,expMax, basis = "weight")
        """

        if qvar == None:
            qvar = self.qvar

        if basis == None:
            warnings.warn("No basis is specified, weight is assumed")
            basis = "weight"


        rho = weyl_vector(["A",1])
        mu = sum([a*(w*rho) for a, w in zip(self.A, wh)])

        delta = self.delta(["A",1])

        def fprefexp(n):
            prefl = list()
            expl = list()
            k = n[0]
            prefl.append((z^(2*k*self.d)-2+z^(-2*k*self.d))
                         * (z-z^(-1))^(-2))
            expl.append(delta+(self.d*self.m*k+mu[0]) ^ 2/(self.d*self.m))
            return prefl, expl

        return expand(q_series(fprefexp, expMax-delta, 1, qvar=qvar))

    def z_hat(self,type_rank,b,expMax, basis="weight"):
        """
        Compute the z_hat invariant.

        INPUT:
            -   ``type_rank`` - [str,int]; Lie group identifier
            -   ``b`` --  List of length self.L of vectors of rank rk; boundary condition
            -   ``expMax`` -- Integer; maximum power of expansion
            -   ``basis`` -- String; basis in which b is given.

        EXAMPLES::

            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: b = S.boundary_conditions(["A",1],basis = "root")[0];
            ....: expMax = 100;
            ....: S.z_hat(["A",1], b, expMax, basis = "root")
            -q^(197/4) + q^(101/4) - q^(5/4) + q^(1/4)

            sage: S = Seifert([0,1,3,1,2,1,6]);
            ....: b = S.boundary_conditions(["A",2],basis = "root")[0];
            ....: expMax = 100;
            ....: S.z_hat(["A",2], b, expMax, basis = "root")
            -2*q^94 - 2*q^92 + 2*q^77 + 2*q^76 + q^65 + 4*q^58 + 2*q^53 - 4*q^50 - 2*q^44 - 2*q^40 + 4*q^32 - 4*q^29 + 2*q^26 - 2*q^22 + q^17 + q^5 + 2*q^4 - 4*q^2 + q
        """
        if basis == None:
            warnings.warn("No basis is specified, weight is assumed")
            basis = "weight"
        
        WG = weyl_group(type_rank)
        if basis == "weight":
            WG = [g.transpose() for g in WG]
        WL = weyl_lengths(type_rank)
        wh_l = list(itertools.product(*[WG if d==1 else [matrix(type_rank[1])] for d in self.deg]))
        wl_l = list(itertools.product(WL,repeat=3))
        Zhat_integrand = 0
        for wh,l in zip(wh_l,wl_l):
            Zhat_integrand += product(l)*self.chi_tilde(type_rank, wh,b,expMax,basis=basis)
        Zhat_integrand *= self._zhat_prefactor(type_rank)
        try:
            return const_term(Zhat_integrand)
        except:
            return Zhat_integrand


"""
References:
    [1] Cheng, M.C., Chun, S., Feigin, B., Ferrari, F., Gukov, S., Harrison, S.M. and Passaro, D., 2022. 3-Manifolds and VOA Characters. arXiv preprint arXiv:2201.04640.
"""
