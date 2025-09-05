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
