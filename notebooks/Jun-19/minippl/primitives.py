"""Deterministic primitive functions shared by FOPPL and HOPPL.

Values at run time:
  numbers (int/float), booleans, strings, None,
  vectors  -> Python lists,
  hash-maps-> Python dicts,
  matrices -> numpy arrays (produced by the mat-* primitives),
  distributions -> minippl.distributions.Distribution instances.

All primitives are *pure functions* of their arguments (put/append/conj etc.
return fresh data; they never mutate), so program states can be shared freely
across particles/traces.
"""

from __future__ import annotations

import math

import numpy as np

from .distributions import DISTRIBUTIONS, Distribution


def _num(x):
    if isinstance(x, bool):
        return 1 if x else 0
    return x


def _add(*args):
    if args and isinstance(args[0], np.ndarray):
        out = args[0]
        for a in args[1:]:
            out = out + a
        return out
    return sum(_num(a) for a in args)


def _sub(*args):
    if len(args) == 1:
        return -_num(args[0])
    out = _num(args[0])
    for a in args[1:]:
        out = out - _num(a)
    return out


def _mul(*args):
    out = _num(args[0])
    for a in args[1:]:
        out = out * _num(a)
    return out


def _div(*args):
    if len(args) == 1:
        return 1.0 / _num(args[0])
    out = _num(args[0])
    for a in args[1:]:
        out = out / _num(a)
    return out


def _eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _num(a) == _num(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return a == b


def _vector(*args):
    return list(args)


def _hash_map(*args):
    if len(args) % 2:
        raise ValueError("hash-map: odd number of arguments")
    return {args[i]: args[i + 1] for i in range(0, len(args), 2)}


def _get(coll, key, default=None):
    if isinstance(coll, dict):
        return coll.get(key, default)
    idx = int(key)
    if isinstance(coll, np.ndarray):
        return coll[idx]
    if 0 <= idx < len(coll):
        return coll[idx]
    return default


def _put(coll, key, value):
    if isinstance(coll, dict):
        out = dict(coll)
        out[key] = value
        return out
    out = list(coll)
    out[int(key)] = value
    return out


def _first(v):
    return v[0]


def _second(v):
    return v[1]


def _last(v):
    return v[-1]


def _rest(v):
    return list(v[1:])


def _nth(v, i):
    return v[int(i)]


def _conj(coll, *xs):
    """conj appends to vectors (lists), like Clojure's conj on vectors."""
    out = list(coll)
    out.extend(xs)
    return out


def _cons(x, coll):
    return [x] + list(coll)


def _append(coll, *xs):
    """(append v x) -> v with x appended (book usage on vectors)."""
    out = list(coll)
    out.extend(xs)
    return out


def _concat(*colls):
    out = []
    for c in colls:
        out.extend(list(c))
    return out


def _count(coll):
    return len(coll)


def _empty(coll):
    return len(coll) == 0


def _peek(coll):
    """Clojure peek on a vector = last element."""
    return coll[-1]


def _range(*args):
    return list(range(*[int(a) for a in args]))


def _to_mat(x):
    return np.asarray(x, dtype=float)


def _mat_mul(a, b):
    return _to_mat(a) @ _to_mat(b)


def _mat_add(a, b):
    return _to_mat(a) + _to_mat(b)


def _mat_transpose(a):
    return _to_mat(a).T


def _mat_tanh(a):
    return np.tanh(_to_mat(a))


def _mat_relu(a):
    return np.maximum(_to_mat(a), 0.0)


def _mat_repmat(a, r, c):
    return np.tile(_to_mat(a), (int(r), int(c)))


PRIMITIVES = {
    # arithmetic
    "+": _add,
    "-": _sub,
    "*": _mul,
    "/": _div,
    "sqrt": lambda x: math.sqrt(_num(x)),
    "exp": lambda x: math.exp(_num(x)),
    "log": lambda x: math.log(_num(x)),
    "pow": lambda x, y: _num(x) ** _num(y),
    "abs": lambda x: abs(_num(x)),
    "floor": lambda x: math.floor(_num(x)),
    "ceil": lambda x: math.ceil(_num(x)),
    "tanh": lambda x: math.tanh(_num(x)),
    "max": lambda *xs: max(_num(x) for x in xs),
    "min": lambda *xs: min(_num(x) for x in xs),
    "mod": lambda a, b: _num(a) % _num(b),
    # comparison / logic (and/or are also special forms in the evaluators;
    # these function versions exist so they can be passed around in HOPPL)
    "=": _eq,
    "==": _eq,
    "!=": lambda a, b: not _eq(a, b),
    "<": lambda a, b: _num(a) < _num(b),
    ">": lambda a, b: _num(a) > _num(b),
    "<=": lambda a, b: _num(a) <= _num(b),
    ">=": lambda a, b: _num(a) >= _num(b),
    "and": lambda *xs: all(bool(x) for x in xs),
    "or": lambda *xs: any(bool(x) for x in xs),
    "not": lambda x: not bool(x),
    # data structures
    "vector": _vector,
    "list": _vector,
    "hash-map": _hash_map,
    "get": _get,
    "put": _put,
    "assoc": _put,
    "first": _first,
    "second": _second,
    "last": _last,
    "rest": _rest,
    "nth": _nth,
    "conj": _conj,
    "cons": _cons,
    "append": _append,
    "concat": _concat,
    "count": _count,
    "empty?": _empty,
    "peek": _peek,
    "range": _range,
    "vector?": lambda x: isinstance(x, list),
    "map?": lambda x: isinstance(x, dict),
    "number?": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    # matrices (Bayesian neural net example, ch. 2.3.3)
    "mat-mul": _mat_mul,
    "mat-add": _mat_add,
    "mat-transpose": _mat_transpose,
    "mat-tanh": _mat_tanh,
    "mat-relu": _mat_relu,
    "mat-repmat": _mat_repmat,
}

# distribution constructors are primitives too
PRIMITIVES.update(DISTRIBUTIONS)


def is_primitive(name: str) -> bool:
    return name in PRIMITIVES
