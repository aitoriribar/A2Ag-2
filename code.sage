from admcycles.admcycles import fundclass, psiclass, lambdaclass, forgetful_pullback, permutation_action
from admcycles.stable_graph import StableGraph
from admcycles.tautological_ring import TautologicalRing
from sage.combinat.integer_vector import IntegerVectors
from sage.combinat.partition import Partition, Partitions
from sage.functions.other import factorial, floor
from sage.misc.misc_c import prod
from sage.rings.integer_ring import ZZ
from sage.groups.perm_gps.permgroup import PermutationGroup
from sage.rings.rational_field import QQ

import itertools

from typing import Sequence, Tuple, Optional

Triple = Tuple[int, int, int]
Quad   = Tuple[int, int, int, int]
Quin   = Tuple[int, int, int, int, int]

def multinomial_coeff(collection):
  r"""
computes the multinomial coefficient of the list [a1,...,an], which equals (a1+...+an)!/a1!...an!
  """
  Num = ZZ(factorial(sum(collection)))
  Den = ZZ(prod(factorial(ai) for ai in collection))
  return Num / Den

def aut_mu(mu):
    r"""
    Calculate the number of automorphisms of the partition mu.

    EXAMPLES::

        sage: from admcycles.gw_adm import aut_mu
        sage: aut_mu([1,1,2,2,2])
        12
        sage: [(P, aut_mu(P)) for P in Partitions(5)]
        [([5], 1),
         ([4, 1], 1),
         ([3, 2], 1),
         ([3, 1, 1], 2),
         ([2, 2, 1], 2),
         ([2, 1, 1, 1], 6),
         ([1, 1, 1, 1, 1], 120)]
    """
    return ZZ(prod(factorial(ei) for ei in Partition(sorted(mu, reverse=True)).to_exp()))

def allowed_monomials_2(h: int):
    """
    Return all tuples (a,b,c,d,e) of integers such that:
      1 <= a <= h
      0 <= b <= h
      0 <= c
      0 <= d
      c + d <= a + b - 1
      e in {0, 1}
      c - a - b + e in {-2, -3}
        need to figure out what happens if b=h and e =1
    """
    if not isinstance(h, int) or h <= 0:
        raise ValueError("h must be a positive integer")

    out = []
    for a in range(1, h + 1):
        for b in range(0, h + 1):
            for e in (0, 1):
                # From c - a - b + e ∈ {-2, -3}  ⇒  c = a + b - e - 2  or  c = a + b - e - 3
                for k in (-2, -3):
                    c = a + b - e + k
                    if c < 0:
                        continue
                    max_d = a + b - 1 - c   # from c + d <= a + b - 1
                    if max_d < 0:
                        continue
                    for d in range(0, max_d + 1):
                        out.append((a, b, c, d, e))
    return out

def allowed_monomials_1(h: int):
    """
    Return all tuples (a,b,c) of integers such that:
      1 <= a <= h
      0 <= b <= h
      0 <= c
      c - a - b in {-2, -3}
    """
    if not isinstance(h, int) or h <= 0:
        raise ValueError("h must be a positive integer")

    out = []
    for a in range(1, h + 1):
        for b in range(0, h + 1):
            for k in (-2, -3):
                # From c - a - b ∈ {-2, -3}  ⇒  c = a + b - e - 2  or  c = a + b - e - 3
                c = a + b + k
                out.append((a, b, c))
    return out

def allowed_monomials_11(h: int):
    """
    Return all tuples (a,b,c,d,e) of integers such that:
      1 <= a <= h
      0 <= b <= h
      0 <= c
      0 <= d
      c + d - a - b  = -2
    """
    if not isinstance(h, int) or h <= 0:
        raise ValueError("h must be a positive integer")

    out = []
    for a in range(1, h + 1):
        for b in range(0, h + 1):
            l = a + b -2
                if l < 0:
                    continue
                for c in range(0, l + 1):
                    out.append((a, b, c, l-c))
    return out

def integral_genus_1_root(triples: Sequence[Triple]):
    r"""
    Compute the integral of lambda1 times a product of psi classes that is obtained from the list

    Parameters
    ----------
    triples : sequence of (int, int, int)
        Example: [(1, 2, 3), (0, 2, 4)]

    Assumptions: all integers are non-negative (>= 0).

    Output
    ------
    the integral \int_{Mbar_{1,k+1}}\lambda_1 prod_{i=1}^k (-psi_i)^{triples[i][0] + triples[i][1] - triples[i][2]-1},

    where k is the length of the list.
    """

    # --- Validate triples ---
    if not isinstance(triples, (list, tuple)):
        raise TypeError("triples must be a list or tuple of 3-int tuples")

    for i, t in enumerate(triples):
        if not (isinstance(t, tuple) and len(t) == 3):
            raise ValueError(f"triples[{i}] must be a tuple of length 3")
        if not all(isinstance(x, int) for x in t):
            raise TypeError(f"triples[{i}] must contain integers only")
        if not all(x >= 0 for x in t):  # change to x > 0 if you want strictly positive
            raise ValueError(f"triples[{i}] must be non-negative")
        length = len(triples)
        exponent_list = [a+b-c-1 for (a,b,c) in triples]
    if sum(exponent_list) == length:
        return ((-1)**length)*(1/24)*multinomial_coeff(triples)
    else:
        return 0

def integral_genus_h_leaf_genus_2_root(vals: Sequence[int], h: int, insertion: bool):
    r"""
    vals: [a, b, c, d, e]
    h: integer > a and > b
    insertion: if False -> subscripts = (h-a, h-b, h) or (h-a, h-b-1, h)
               if True  -> subscripts = (1, h-a, h-b, h) or (1, h-a, h-b-1, h)
    Returns The integral of psi^c * prod_i lambda_{subscripts[i]} times a combinatorial factor
    """
    if len(vals) < 2:
        raise ValueError("vals must have at least two elements (a and b).")
    a, b, c, d, e = vals[0], vals[1], vals[2], vals[3], vals[4]
    if not all(isinstance(x, int) for x in (a, b, h)):
        raise TypeError("a, b, and h must be integers.")
    if a < 0 or b < 0 or h < max(a, b):
        raise ValueError("Require non-negative a,b and h > a and h > b.")
    if e=0:
        if not insertion:
            subscripts = (h - a, h - b, h)
        if insertion:
            subscripts = (1, h - a, h - b, h)
    if e=1:
        if not insertion:
            subscripts = (h - a, h - b - 1, h)
        if insertion:
            subscripts = (1, h - a, h - b - 1, h)

    counts = tuple([0] * h)  # indices 0..h-1 correspond to values 1..h
    for s in subscripts:
        if 1 <= s <= h:          # only count valid positions
            counts[s - 1] += 1
    if e=0:
        return (-1)**(a+b)*multinomial_coeff((c,d))*hodge_integral(h,1,counts, (), (c,))
    if e=1:
        return (b+1)*(-1)**(a+b)*multinomial_coeff((c,d))*hodge_integral(h,1,counts, (), (c,))

def integral_genus_h_leaf_genus_1_1_root(vals: Sequence[int], h: int, insertion: bool) -> List[int]:
    r"""
    vals: [a, b, c, d]
    h: integer > a and > b
    insertion: if False -> subscripts = (h-a, h-b, h) 
               if True  -> subscripts = (1, h-a, h-b, h) 
    Returns The integral \int_{Mbar_{h,2}} psi1^c * psi2^d * prod_i lambda_{subscripts[i]} times a combinatorial factor
    """
    if len(vals) < 4:
        raise ValueError("vals must have at least four elements (a, b, c, d).")
    a, b, c, d = vals[0], vals[1], vals[2], vals[3]
    if not all(isinstance(x, int) for x in (a, b, c, d, h)):
        raise TypeError("a, b, and h must be integers.")
    if a < 0 or b < 0 or c <0 or d < 0 or h < max(a, b):
        raise ValueError("Require non-negative a,b,c,d and h >= a and h >= b.")
    if not insertion:
        subscripts = (h - a, h - b, h)
    if insertion:
        subscripts = (1, h - a, h - b, h)
    counts = tuple([0] * h)  # indices 0..h-1 correspond to values 1..h
    for s in subscripts:
        if 1 <= s <= h:          # only count valid positions
            counts[s - 1] += 1
    return (-1)**(a+b)*hodge_integral(h,1,counts, (), (c,d))

def integral_genus_h_leaf_genus_1_root(vals: Sequence[int], h: int, insertion: bool) -> List[int]:
    r"""
    vals: [a, b, c]
    h: integer > a and > b
    insertion: if False -> subscripts = (h-a, h-b, h) 
               if True  -> subscripts = (1, h-a, h-b, h) 
    Returns The integral \int_{Mbar_{h,2}} psi1^c * psi2^d * prod_i lambda_{subscripts[i]} times a combinatorial factor
    """
    if len(vals) < 4:
        raise ValueError("vals must have at least four elements (a, b, c, d).")
    a, b, c, d = vals[0], vals[1], vals[2], vals[3]
    if not all(isinstance(x, int) for x in (a, b, c, d, h)):
        raise TypeError("a, b, and h must be integers.")
    if a < 0 or b < 0 or c <0 or d < 0 or h < max(a, b):
        raise ValueError("Require non-negative a,b,c,d and h >= a and h >= b.")
    if not insertion:
        subscripts = (h - a, h - b, h)
    if insertion:
        subscripts = (1, h - a, h - b, h)
    counts = tuple([0] * h)  # indices 0..h-1 correspond to values 1..h
    for s in subscripts:
        if 1 <= s <= h:          # only count valid positions
            counts[s - 1] += 1
    return (-1)**(a+b)*hodge_integral(h,1,counts, (), (c,d))

def integral_genus_1_root(quintuples: Sequence[Quin]):
    r"""
    Compute the integral of lambda1 times a product of psi classes that is obtained from the list

    Parameters
    ----------
    quintuples : sequence of (int, int, int, int, int)
        Example: [(1, 2, 3, 1, 0), (0, 2, 4, 2, 1)]

    Assumptions: all integers are non-negative (>= 0).

    Output
    ------
    the integral \int_{Mbar_{1,k+1}}\lambda_1 prod_{i=1}^k (-psi_i)^{quintuples[i][0] + quintuples[i][1] - quintuples[i][2]-1},

    where k is the length of the list.
    """

    # --- Validate triples ---
    if not isinstance(quintuples, (list, tuple)):
        raise TypeError("quintuples must be a list or tuple of 5-int tuples")

    for i, t in enumerate(quintuples):
        if not (isinstance(t, tuple) and len(t) == 5):
            raise ValueError(f"quintuples[{i}] must be a tuple of length 5")
        if not all(isinstance(x, int) for x in t):
            raise TypeError(f"quintuples[{i}] must contain integers only")
        if not all(x >= 0 for x in t):  # change to x > 0 if you want strictly positive
            raise ValueError(f"triples[{i}] must be non-negative")
    k = len(quintuples)
    exponent_list = [a+b-c-1 for (a,b,c) in quintuples]
    if sum(exponent_list) == k:
        return ((-1)**length)*(1/24)*multinomial_coeff(exponent_list)
    else:
        return 0

def integral_genus_2_root(quintuples: Sequence[Quin], insertion: bool):
    r"""
    Compute the integral of lambda2 times a product of psi/boundary classes that is obtained from the list

    Parameters
    ----------
    quintuples : sequence of (int, int, int, int, int)
        Example: [(1, 2, 3, 1, 0), (0, 2, 4, 2, 1)]

    Assumptions: all integers are non-negative (>= 0).

    Output
    ------
    the integral \int_{Mbar_{2,k}}\lambda_3 prod_{i=1}^k (-psi_i)^{triples[i][0] + triples[i][1] - triples[i][2]-1},

    where k is the length of the list.
    """

    # --- Validate quintuples ---
    if not isinstance(quintuples, (list, tuple)):
        raise TypeError("quintuples must be a list or tuple of 5-int tuples")

    for i, t in enumerate(quintuples):
        if not (isinstance(t, tuple) and len(t) == 5):
            raise ValueError(f"quintuples[{i}] must be a tuple of length 5")
        if not all(isinstance(x, int) for x in t):
            raise TypeError(f"quintuples[{i}] must contain integers only")
        if not all(x >= 0 for x in t):  # change to x > 0 if you want strictly positive
            raise ValueError(f"triples[{i}] must be non-negative")
    k = len(quintuples)
    if insertion:
        lambda1insertions = sum(quintuples[i][4] for i in range(0, k)) +1
    else:
        lambdainsertions = sum(quintuples[i][4] for i in range(0, k))
    if lambdainsertions > 1:
        return 0
    if lambdainsertions == 1:
        integrand = lambdaclass(1,2,k) * lambdaclass(2,2,k)
        intergand_codimension = 3
    if lambdainsertions == 0:
        integrand = lambdaclass(1,2,k)
        integrand_codimension = 2
    for i = 1..k:
        integrand_codimension += quintuple[0] + quintuple[1]-quintuple[2]-1
    if not integrand_codimension == 3 + k:
        return 0
    else:
        G = SymmetricGroup(k)
        S = psiclass(1,2,1)
        S1 = S.forgetful_pullback(range(2, k+1))
        for i = 1..k:
            g = G("1{i}")
            S2 = S1.permutation_action(g) #this is the forgetfull pullback of psi
            integrand = integrand*((S2-psiclass(i,2,k))**(quintuple[0] + quintuple[1]-quintuple[2]-quintuple[3]-1))*(S2**quintuple[3])
        return integral.evaluate()

def ordered_set_decompositions(S, m):
    r"""
    Generate all lists

    [S1, ..., Sm]

    of ways to decompose the set S into m subsets.

    EXAMPLES::

        sage: from admcycles.gw_adm import ordered_set_decompositions
        sage: list(ordered_set_decompositions({2,4,7}, 2))
        [[{2, 4, 7}, set()],
         [{2, 4}, {7}],
         [{2, 7}, {4}],
         [{2}, {4, 7}],
         [{4, 7}, {2}],
         [{4}, {2, 7}],
         [{7}, {2, 4}],
         [set(), {2, 4, 7}]]
        sage: len(list(ordered_set_decompositions({-12,4,7,9}, 5))) == 5^4
        True
    """
    for di in itertools.product(*[range(m) for s in S]):
        yield [set([s for j, s in enumerate(S) if di[j] == i]) for i in range(m)]


def star_shaped_graphs(g, n, muvec, h=0, d=None):
    r"""
    Return list over star-shaped graphs of genus g with
    n free marked points, associated to degree d covers of curves
    of genus h with marked ramification profiles

    muvec = (mu_1, ..., mu_m)

    according to [NS25, Section 2.5] subject to the conditions from
    [NS25, Theorem 2.7].

    Output format:

    ((g0, muvec), (g1, eta1, N1), ..., (gk, etak, Nk))

    with the list of tuples (gi, etai, Ni) ordered.
    Here etai is a Partition and Ni is an ordered tuple of markings
    from m+1, ..., m+n, where m is the number of points with relative
    conditions, which all go to v0.

    EXAMPLES::

        sage: from admcycles.gw_adm import star_shaped_graphs
        sage: star_shaped_graphs(2, 0, [[1], [1]], d=1)
        [((0, ((1,), (1,))), (2, [1], ())),
         ((0, ((1,), (1,))), (1, [1], ()), (1, [1], ()))]
        sage: [len(list(star_shaped_graphs(g, 0, [], d=1))) == len(Partitions(g)) for g in range(2,9)]
        [True, True, True, True, True, True, True]
        sage: star_shaped_graphs(2, 0, [[2], [2]])
        [((2, ((2,), (2,))), (0, [2], ()), (0, [2], ()), (0, [2], ()), (0, [2], ())),
         ((1, ((2,), (2,))), (0, [2], ()), (1, [2], ())),
         ((1, ((2,), (2,))), (0, [1, 1], ()), (0, [2], ()), (0, [2], ())),
         ((0, ((2,), (2,))), (1, [1, 1], ())),
         ((0, ((2,), (2,))), (0, [1, 1], ()), (0, [1, 1], ()))]
    """
    if d is None:
        if not muvec:
            raise ValueError(
                'For empty vector of relative conditions, degree d must be specified manually')
        d = ZZ(sum(muvec[0]))
    etalist = [list(Partitions(d, length=ell)) for ell in range(d + 1)]
    etanums = [len(Pi) for Pi in etalist]
    muvectuple = tuple([tuple(mui) for mui in muvec])
    m = sum(len(mui) for mui in muvec)
    resultlist = []

    for g0 in range(g + 1):
        # g0 = euler char 2g_0 -2 + n(v_0) of root vertex
        g0 = ZZ(g0)
        b = ZZ(2) * g0 - 2 - d * (2 * h - 2) - sum(d - len(mui)
                                                   for mui in muvec)  # remaining degree of the branch divisor

        for bvec in Partitions(b, max_part=d - 1):
            # bvec = vector of POSITIVE numbers d-ell(eta^i)
            evec = bvec.to_exp()
            # append some zeros for ensuring evec has d-1 entries
            evec = evec + [ZZ(0) for i in range(d - 1 - len(evec))]
            # d=5, evec = (1,0,2,8) means that bvec = (1^1 3^2 4^8)

            # before continuing, we must also take into account how many additional
            # vectors eta_i we can add of the form eta_i = (1,1,...,1)
            if d == 1:
                # for d = 1 the vector eta_i = (1) is only allowed if g_i >=1 or n_i >= 1
                eta0_max = g - g0 + n
            else:
                eta0_max = floor(
                    (g - g0 - sum(d - bi - 1 for bi in bvec)) / (d - ZZ(1)))
                # maximum that can satisfy g = sum(g_i + ell(eta^i)-1) + g_0
            for eta0 in range(eta0_max + 1):
                etilde = [eta0] + list(evec)
                # d=5, etilde = (3,1,0,2,8) means that bvec = (0^3 1^1 3^2 4^8)
                # etilde has length d
                assert len(etilde) == d
                for etaindices in itertools.product(*[IntegerVectors(ei, etanums[d - i]) for i, ei in enumerate(etilde)]):
                    # at this point g0 and eta_1, ..., eta_k are determined
                    # it remains to distribute the genera g_i and markings n_i to the outlying vertices
                    # we do this in a lazy way and use ordering and sets to eliminate duplicates
                    etavec = [etalist[d - i][j] for i, ei in enumerate(
                        etaindices) for j, multj in enumerate(ei) for mi in range(multj)]
                    k = len(etavec)
                    g_remaining = g - g0 - \
                        sum(len(etai) -
                            1 for etai in etavec)  # = sum_{i=1}^k g_i
                    for gvec in IntegerVectors(g_remaining, k):
                        for Nvec in ordered_set_decompositions(set(range(m + 1, m + n + 1)), k):
                            cand_vect = tuple((gvec[i], etavec[i], tuple(
                                sorted(Nvec[i]))) for i in range(k))
                            if not all(2 * gi - 2 + d + len(etai) + len(Ni) > 0 for gi, etai, Ni in cand_vect):
                                continue
                            resultlist.append(
                                ((g0, muvectuple), *sorted(cand_vect)))
    return sorted(set(resultlist), reverse=True)


def aut_star_shaped_graph(gamma):
    r"""
    Calculate number of automorphisms of tuple describing a star shaped graph
    (convention according to the function star_shaped_graphs).

    EXAMPLES::

        sage: from admcycles.gw_adm import star_shaped_graphs, aut_star_shaped_graph
        sage: for gamma in star_shaped_graphs(3, 0, [[2], [2]]):
        ....:     print(f'Graph partition: {gamma}')
        ....:     print(f'Automorphism number : {aut_star_shaped_graph(gamma)}')
        ....:
        Graph partition: ((3, ((2,), (2,))), (0, [2], ()), (0, [2], ()), (0, [2], ()), (0, [2], ()), (0, [2], ()), (0, [2], ()))
        Automorphism number : 720
        Graph partition: ((2, ((2,), (2,))), (0, [2], ()), (0, [2], ()), (0, [2], ()), (1, [2], ()))
        Automorphism number : 6
        Graph partition: ((2, ((2,), (2,))), (0, [1, 1], ()), (0, [2], ()), (0, [2], ()), (0, [2], ()), (0, [2], ()))
        Automorphism number : 48
        Graph partition: ((1, ((2,), (2,))), (1, [2], ()), (1, [2], ()))
        Automorphism number : 2
        Graph partition: ((1, ((2,), (2,))), (0, [2], ()), (2, [2], ()))
        Automorphism number : 1
        Graph partition: ((1, ((2,), (2,))), (0, [2], ()), (0, [2], ()), (1, [1, 1], ()))
        Automorphism number : 4
        Graph partition: ((1, ((2,), (2,))), (0, [1, 1], ()), (0, [2], ()), (1, [2], ()))
        Automorphism number : 2
        Graph partition: ((1, ((2,), (2,))), (0, [1, 1], ()), (0, [1, 1], ()), (0, [2], ()), (0, [2], ()))
        Automorphism number : 16
        Graph partition: ((0, ((2,), (2,))), (2, [1, 1], ()))
        Automorphism number : 2
        Graph partition: ((0, ((2,), (2,))), (0, [1, 1], ()), (1, [1, 1], ()))
        Automorphism number : 4
        Graph partition: ((0, ((2,), (2,))), (0, [1, 1], ()), (0, [1, 1], ()), (0, [1, 1], ()))
        Automorphism number : 48
    """
    aut = ZZ(1)
    m = len(gamma)
    if m == 1:
        return aut
    old = gamma[1]
    count = 1
    for j in range(2, m + 1):
        if j < m and gamma[j] == old:
            count += 1
        else:
            aut *= factorial(count) * aut_mu(old[1]) ** count
            if j < m:
                old = gamma[j]
                count = 1
    return aut


def disconnected_partitions(g, e, P):
    r"""
    Enumerates data of all moduli spaces of possibly disconnected curves
    of arithmetic genus g with markings 1, ..., e associated to half-edges
    and markings P associated to free markings (assumed larger than e).

    Here the genus condition is

    \sum_{i=1}^ell (g_i-1) + 1 = g.

    An additional condition is that each component must contain at least one of
    the markings 1, ..., e.

    EXAMPLES::

        sage: from admcycles.gw_adm import disconnected_partitions
        sage: disconnected_partitions(1, 2, [])
        [[(1, (1, 2))],
         [(2, (1,)), (0, (2,))],
         [(1, (1,)), (1, (2,))],
         [(0, (1,)), (2, (2,))]]
    """
    N = set(list(range(1, e + 1)) + list(P))
    result = []
    for Spart in SetPartitions(N):
        if any(all(si > e for si in Si) for Si in Spart):
            # one of the sets contains no half-edge, only free markings
            continue
        Spartlist = list(Spart)
        ell = len(Spartlist)
        gsum = g + ell - ZZ(1)
        for gvec in IntegerVectors(gsum, ell):
            result.append([(gi, tuple(sorted(Ni)))
                          for gi, Ni in zip(gvec, Spartlist)])
    return result


def stable_star_graph(graph_partition, outlying_vertices):
    r"""
    Given a partition describing a star-shaped graph (as from ``star_shaped_graphs``)
    and a list of partitions for its outlying vertices (as from ``discnonected_partitions``)
    return a tuple:

      * gamma : Stable graph associated to these partitions; the 0-th vertex is the root
      * etadict : dictionary sending half-edges of gamma to ramification orders
      * out_half_edges: list of lists of half-edges on central vertex going to each of the (possibly disconnected)
        outlying vertex groups; has the same length as outlying_vertices
        this a priori contains half-edges that were forgotten when stabilizing the graph gamma
      * out_vertices: list of lists of outlying vertices (that are stable)

    EXAMPLES::

        sage: from admcycles.gw_adm import stable_star_graph, star_shaped_graphs, disconnected_partitions
        sage: import itertools
        sage: for graph_partition in star_shaped_graphs(2, 1, [[2], [2]]):
        ....:     for outlying_vertices in itertools.product(*[disconnected_partitions(gi, len(etai), Ni) for gi, etai, Ni in graph_partition[1:]]):
        ....:         gamma, etadict, out_half_edges, out_vertices = stable_star_graph(graph_partition, outlying_vertices)
        ....:         print(graph_partition)
        ....:         print(outlying_vertices)
        ....:         print(gamma)
        ....:         print(etadict)
        ....:         print(out_half_edges)
        ....:         print(out_vertices)
        ....:         print('')
        ....:
        ((2, ((2,), (2,))), (0, [2], ()), (0, [2], ()), (0, [2], ()), (0, [2], (3,)))
        ([(0, (1,))], [(0, (1,))], [(0, (1,))], [(0, (1, 3))])
        [2] [[1, 2, 3]] []
        {4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 3: 2, 11: 2}
        [[4], [6], [8], [3]]
        [[], [], [], []]
        <BLANKLINE>
        ((1, ((2,), (2,))), (0, [2], (3,)), (1, [2], ()))
        ([(0, (1, 3))], [(1, (1,))])
        [1, 1] [[1, 2, 6, 3], [7]] [(6, 7)]
        {3: 2, 5: 2, 6: 2, 7: 2}
        [[3], [6]]
        [[], [1]]
        <BLANKLINE>
        ((1, ((2,), (2,))), (0, [2], ()), (1, [2], (3,)))
        ([(0, (1,))], [(1, (1, 3))])
        [1, 1] [[1, 2, 6], [7, 3]] [(6, 7)]
        {4: 2, 5: 2, 6: 2, 7: 2}
        [[4], [6]]
        [[], [1]]
        <BLANKLINE>
        ((1, ((2,), (2,))), (0, [1, 1], (3,)), (0, [2], ()), (0, [2], ()))
        ([(0, (1, 2, 3))], [(0, (1,))], [(0, (1,))])
        [1, 0] [[1, 2, 4, 6], [5, 7, 3]] [(4, 5), (6, 7)]
        {4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2}
        [[4, 6], [8], [10]]
        [[1], [], []]
        ...
        ((0, ((2,), (2,))), (0, [1, 1], ()), (0, [1, 1], (3,)))
        ([(0, (1,)), (1, (2,))], [(0, (1, 3)), (1, (2,))])
        [0, 1, 1] [[1, 2, 6, 10, 3], [7], [11]] [(6, 7), (10, 11)]
        {4: 1, 5: 1, 6: 1, 7: 1, 3: 1, 9: 1, 10: 1, 11: 1}
        [[4, 6], [3, 10]]
        [[1], [2]]
        <BLANKLINE>
        ((0, ((2,), (2,))), (0, [1, 1], ()), (0, [1, 1], (3,)))
        ([(0, (1,)), (1, (2,))], [(1, (1,)), (0, (2, 3))])
        [0, 1, 1] [[1, 2, 6, 8, 3], [7], [9]] [(6, 7), (8, 9)]
        {4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 3: 1, 11: 1}
        [[4, 6], [8, 3]]
        [[1], [2]]
        <BLANKLINE>
        ((0, ((2,), (2,))), (0, [1, 1], ()), (0, [1, 1], (3,)))
        ([(0, (1,)), (1, (2,))], [(0, (1,)), (1, (2, 3))])
        [0, 1, 1] [[1, 2, 6, 10], [7], [11, 3]] [(6, 7), (10, 11)]
        {4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}
        [[4, 6], [8, 10]]
        [[1], [2]]
        <BLANKLINE>
    """
    if len(graph_partition) != len(outlying_vertices) + 1:
        raise ValueError(
            'length of graph_partition must be precisely one larger than length of outlying_vertices')
    markings = sum([Ni for _, _, Ni in graph_partition[1:]], ())
    muvec = graph_partition[0][1]
    m = sum(len(mui) for mui in muvec)
    N = max(markings + (m,))  # edges can start at (N+1, N+2), ...
    genera = [graph_partition[0][0]]  # central vertex
    legs = [list(range(1, m + 1))]  # markings for relative conditions on v0
    edges = []
    count = N + 1  # running count for half-edge indices
    out_half_edges = []
    out_vertices = []
    etadict = dict()

    for (gi, etai, Ni), Vout in zip(graph_partition[1:], outlying_vertices):
        old_genera = len(genera)
        genera += [gi for gi, _ in Vout]  # genera done
        out_vertices.append(list(range(old_genera, old_genera + len(Vout))))
        newlegs = [[(pi if pi in Ni else 2 * (pi - 1) + count + 1)
                    for pi in P] for _, P in Vout]
        legs += newlegs
        new_out_half_edges = []
        for i, ram in enumerate(etai):
            edges.append((count, count + 1))  # edges done
            etadict[count] = ram
            etadict[count + 1] = ram  # etadict done
            legs[0].append(count)
            new_out_half_edges.append(count)
            count += 2
        out_half_edges.append(new_out_half_edges)
        # legs done

    gamma0 = StableGraph(genera, legs, edges)
    gamma = gamma0.copy()
    dicv, dicl, _ = gamma.stabilize()
    # sends forgotten half-edges to markings that replaced them
    diclinv = {b: a for a, b in dicl.items()}

    # annoying relabeling in etadict and out_half_edges from contracted components
    etadict = {diclinv.get(a, a): b for a, b in etadict.items()}
    out_half_edges = [[diclinv.get(a, a) for a in ohe]
                      for ohe in out_half_edges]
    out_vertices = [[v for v in dicv if dicv[v] in ov] for ov in out_vertices]

    return gamma, etadict, out_half_edges, out_vertices


def hurwitz_cycle(g, muvec, etavec, remembered_markings, d=None, gamma=[], alpha=None):
    r"""
    Implement the Hurwitz cycle from [NS25, Section 2.3] in the cases
    where we know it.

        EXAMPLES::

        sage: from admcycles.gw_adm import hurwitz_cycle
    """
    if gamma or alpha is not None:
        raise NotImplementedError(
            'Insertions from the target are not implemented yet')
    if (not muvec) and (not etavec) and d is None:
        raise ValueError(
            'For empty ramification lists the degree d must be specified manually')
    if d is None:
        d = sum((list(muvec) + list(etavec))[0])

    m = sum(len(mui) for mui in muvec)
    n = sum(len(etai) for etai in etavec)
    if d == 1:
        return fundclass(g, m + n) if g == 0 else TautologicalRing(g, m + n).zero()
    if d == 2:
        G = PermutationGroup([(1, 2)])
        HDat = []
        n_weierstrass = 0
        n_hyperell = 0
        for etai in list(muvec) + list(etavec):
            if len(etai) == 1:
                HDat.append(G[1])
                n_weierstrass += 1
            elif len(etai) == 2:
                HDat.append(G[0])
                n_hyperell += 1
            else:
                raise ValueError('invalid etai')
        # fill up with missing Weierstrass points
        HDat += [G[1] for i in range(2 * g + 2 - n_weierstrass)]
        H = HurData(G, HDat)
        # / factorial(n_hyperell) / (ZZ(2)**n_hyperell)
        factor = QQ(1) / factorial(2 * g + 2 - n_weierstrass)
        result = factor * Hidentify(g, H, markings=remembered_markings)

        # currently, the markings on result are named 1,2,..,n, 2*g+3, ..., 2*g+2+2*m
        # shift the last set of markings down by 2*g+3-(n+1)
        # rndict = {i: i for i in range(1, n + 1)}
        # rndict.update({j: j - (2 * g + 2 - n) for j in range(2 * g + 2 + 1, 2 * g + 2 + 1 + 2 * m)})
        rndict = {marki: i + 1 for i, marki in enumerate(remembered_markings)}
        return result.rename_legs(rndict, inplace=False)
    raise NotImplementedError('only small degrees are supported for now')


def hurwitz_cycle_dimension(g, muvec, etavec, remembered_markings, d=None, gamma=[], alpha=None):
    r"""
    Implement the complex dimension of the Hurwitz cycle from [NS25, Section 2.3] in the cases
    where we know it.

        EXAMPLES::

        sage: from admcycles.gw_adm import hurwitz_cycle
    """
    if gamma or alpha is not None:
        raise NotImplementedError(
            'Insertions from the target are not implemented yet')
    if (not muvec) and (not etavec) and d is None:
        raise ValueError(
            'For empty ramification lists the degree d must be specified manually')
    if d is None:
        d = sum((list(muvec) + list(etavec))[0])

    m = sum(len(mui) for mui in muvec)
    n = sum(len(etai) for etai in etavec)
    if d == 1:
        return ZZ(3) * g - 3 + m + n if g == 0 else -ZZ(1)
    if d == 2:
        n_hyperell = 0
        for etai in list(muvec) + list(etavec):
            if len(etai) == 2:
                n_hyperell += 1
        return ZZ(2) * g + 2 - 3 + n_hyperell
    raise NotImplementedError('only small degrees are supported for now')


def gw_sum(g, muvec, h=0, gamma=[], d=None, alpha=None, termsout=False):
    r"""
    Return the right-hand side formula from [NS25, Theorem 2.7].

    EXAMPLES::

        sage: from admcycles.gw_adm import gw_sum
        sage: from admcycles import DR_cycle
        sage: muvec = [[1], [1]]
        sage: gw_sum(1, muvec) == DR_cycle(1, [1,-1])
        True
        sage: gw_sum(2, muvec) == DR_cycle(2, [1,-1])
        True
        sage: gw_sum(3, muvec) == DR_cycle(3, [1,-1])
        True
        sage: muvec = [[2], [2]]
        sage: gw_sum(1, muvec) == DR_cycle(1, [2, -2])
        True
        sage: gw_sum(2, muvec) == DR_cycle(2, [2, -2])
        True
    """
    m = len(muvec)
    M = sum(len(mui) for mui in muvec)
    n = len(gamma)
    if gamma or alpha is not None:
        raise NotImplementedError(
            'Insertions from the target are not implemented yet')
    if (h, m) == (0, 1):
        raise ValueError('(h,m) = (0,1) is excluded in the theorem')
    if (not muvec) and d is None:
        raise ValueError(
            'For empty ramification lists the degree d must be specified manually')
    if d is None:
        d = sum(muvec[0])
    # first M markings from muvec, last n markings from gamma
    R = TautologicalRing(g, M + n)
    terms = []  # list of lists [graph_partition, outlying_vertices, contrib]

    for graph_partition in star_shaped_graphs(g, n, muvec, h=h, d=d):
        aut = aut_star_shaped_graph(graph_partition)
        etavec = [etai for _, etai, _ in graph_partition[1:]]
        ell = len(etavec)  # number of outlying vertex groups
        for outlying_vertices in itertools.product(*[disconnected_partitions(gi, len(etai), Ni) for gi, etai, Ni in graph_partition[1:]]):
            Gamma, etadict, out_half_edges, out_vertices = stable_star_graph(
                graph_partition, outlying_vertices)
            ram_degs = [[[etadict[h] for h in Gamma.legs(
                v) if h > M + n] for v in out_vertices[i]] for i in range(ell)]
            contrib = R.zero()
            remembered_markings = list(range(1, M + 1))
            ohe = [a for oh in out_half_edges for a in oh]
            # remembered_markings += [M+1+i for i, he in enumerate(ohe) if he in Gamma.legs(0)]
            remembered_markings += [M + 1 + i for i,
                                    he in enumerate(ohe) if he in Gamma.legs(0)]
            actual_out_half_edges = [
                [he for he in oh if he in Gamma.legs(0)] for oh in out_half_edges]
            g0 = Gamma.genera(0)
            n0 = Gamma.num_legs(0)
            assert len(remembered_markings) == n0
            # H0 = hurwitz_cycle(g0, muvec, etavec, remembered_markings, d=d, gamma=gamma, alpha=alpha)
            # deg_list = H0.degree_list()
            # if not deg_list:
            # print('empty hurwitz cycle')
            # print(graph_partition)
            # print(outlying_vertices)
            # print(g0, muvec, etavec, remembered_markings)
            # continue
            # dimension of cycle on central vertex
            dim0 = hurwitz_cycle_dimension(
                g0, muvec, etavec, remembered_markings, d=d, gamma=gamma, alpha=alpha)
            # add extra info to rule out placing psi-classes on entirely unstable outlying vertices
            # i.e. those with g=0, no marking
            outer = [(dim0 + 1) * ZZ((gi > 0) or (len(etai) > 1) or bool(Ni))
                     for gi, etai, Ni in graph_partition[1:]]
            psi0_vecs = (psi0_vec for e in range(dim0 + 1)
                         for psi0_vec in IntegerVectors(e, ell, outer=outer))
            for psi0_vec in psi0_vecs:
                # insertion at the central vertex
                # print(f'psi0_vec : {psi0_vec}')

                # print('H0 cycle: ')
                # print(H0)

                # for each outlying vertex group, we now want to sum over all contributions
                # having term z^psi_0vec[j] inside I_{g_j, n_j, eta_j}(z)
                # first we record the exponent of z that comes in independent of choices
                # and the coefficient coming from unstable vertices
                e0_vec = []
                unst_coeff = ZZ(1)
                for i, ov in enumerate(outlying_vertices):
                    etai = graph_partition[i + 1][1]
                    # dictionary sending markings in outlying_vertices to eta-values
                    etai = {k + 1: ZZ(ramk) for k, ramk in enumerate(etai)}
                    ev = ZZ(1)  # factor z^1 in equation (2.5)
                    for gk, Pk in ov:
                        # print(outlying_vertices, gk, Pk, etai)
                        if gk == 0 and len(Pk) <= 2:
                            # (g,n) = (0,0), (0,1) case
                            if sum(1 for p in Pk if p in etai) == 1:
                                # single edge going to the central vertex
                                dv = etai[Pk[0]]
                                # ignore the marking corresponding to an edge
                                nv = len(Pk) - 1
                                ev += 1 - dv - nv
                                # print(ev)
                                unst_coeff *= dv**(dv - 1 + nv) / factorial(dv)
                            else:
                                # two edges going to central vertex, no marking
                                etav = [etai[uw] for uw in Pk]
                                dv = sum(etav)
                                ev += -dv
                                unst_coeff *= etav[0]**(etav[0] + 1) * etav[1]**(
                                    etav[1] + 1) / factorial(etav[0]) / factorial(etav[1]) / dv
                        else:
                            # looking at a stable vertex
                            dv = sum(etai.get(www, 0) for www in Pk)
                            ev += -dv - 1 + gk
                    e0_vec.append(ev)
                # at this point in the code e0_vec and unst_coeff are finished
                # print(f'e0_vec : {e0_vec}')
                remaining_degrees = [b - a for a, b in zip(psi0_vec, e0_vec)]
                # print(f'remaining_degrees : {remaining_degrees}')
                if any(rd < 0 for rd in remaining_degrees):
                    continue
                outers = [[3 * Gamma.genera(v) - 3 + Gamma.num_legs(v)
                           for v in out_vertices[i]] for i in range(ell)]
                deg_parts = [IntegerVectors(remaining_degrees[i], len(
                    out_vertices[i]), outer=o) for i, o in enumerate(outers)]

                for deg_part in itertools.product(*deg_parts):
                    # deg_part[i] = [di_1, ..., di_l] means the i-th outlying vertex group
                    # has l stable vertices v_u (from out_vertices[i]) and the lambda-psi-insertion
                    # at v_u has total degree precisely di_u
                    degs = [di for dp in deg_part for di in dp]
                    out_data = [(v, Gamma.genera(v), rdegs) for i, ov in enumerate(
                        out_vertices) for v, rdegs in zip(ov, ram_degs[i])]
                    assert all(out_data[v][0] == v +
                               1 for v in range(Gamma.num_verts() - 1))
                    assert len(degs) == len(out_data)
                    ievecs = [IntegerVectors(di, 1 + len(rdegsi), outer=[g] + [di for _ in rdegsi])
                              for di, (_, gi, rdegsi) in zip(degs, out_data)]
                    for ievec in itertools.product(*ievecs):
                        if not any(ei for ei in psi0_vec):
                            Ins0 = H0 = hurwitz_cycle(g0, muvec, etavec, remembered_markings, d=d, gamma=gamma, alpha=alpha)
                        else:
                            b_orig = sum(len(mui) for mui in muvec) + sum(len(etai) for etai in etavec)
                            extra_markings = [i for i in range(1, b_orig + 1) if i not in remembered_markings]
                            all_markings = remembered_markings + extra_markings
                            H0 = hurwitz_cycle(
                                g0, muvec, etavec, all_markings, d=d, gamma=gamma, alpha=alpha)
                            Ins0 = ZZ(1)
                            for i, ei in enumerate(psi0_vec):
                                if ei == 0:
                                    continue
                                out_he = actual_out_half_edges[i][0]
                                psitild = etadict[out_he] * \
                                    psiclass(Gamma.legs(0).index(
                                        out_he) + 1, g0, b_orig)
                                Ins0 *= (-psitild) ** ei
                            Ins0 *= H0
                            Ins0 = Ins0.forgetful_pushforward(list(range(len(remembered_markings) + 1, b_orig + 1)))

                        insertions = [Ins0]
                        for ie, (v, gi, rdegsi) in zip(ievec, out_data):
                            ni = Gamma.num_legs(v)
                            Ins = (-1)**ie[0] * lambdaclass(ie[0], gi, ni)
                            Ins *= prod(rdegsi[i]**(rdegsi[i] + 1 + expo) / factorial(
                                rdegsi[i]) * psiclass(i + 1, gi, ni)**expo for i, expo in enumerate(ie[1:]))
                            insertions.append(Ins)

                        contrib += 1 / aut * unst_coeff * \
                            Gamma.boundary_pushforward(insertions)

            terms.append([graph_partition, outlying_vertices, contrib])

    if termsout:
        return terms
    return R.sum(contrib for _, _, contrib in terms)  # TODO: replace
