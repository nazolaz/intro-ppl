"""S-expression reader for FOPPL/HOPPL programs.

Forms are represented as plain Python data:
  - Symbol (a str subclass)        identifiers:  x, +, sample, mat-mul
  - int / float                    numbers
  - bool                           true / false
  - str                           "double-quoted strings"
  - list                           compound forms: (op e1 e2 ...)

Square brackets are read as ordinary lists, i.e. ``[x 1]`` == ``(x 1)``;
the FOPPL/HOPPL ``let`` desugarer interprets the binding list itself.
Comments run from ``;`` to end of line.
"""

from __future__ import annotations


class Symbol(str):
    """An identifier. Subclasses str so it is hashable and comparable."""

    __slots__ = ()

    def __repr__(self) -> str:  # print without quotes
        return str(self)


class _String(str):
    """Internal marker so the tokenizer can distinguish "x" from the symbol x."""

    __slots__ = ()


def tokenize(text: str) -> list:
    tokens: list = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\n\r,":
            i += 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "()[]":
            tokens.append("(" if c in "([" else ")")
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(text[j])
                j += 1
            if j >= n:
                raise SyntaxError("unterminated string literal")
            tokens.append(_String("".join(buf)))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\n\r,()[];"':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _atom(token):
    if isinstance(token, _String):
        return str(token)
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "nil":
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return Symbol(token)


def _read(tokens: list, pos: int):
    if pos >= len(tokens):
        raise SyntaxError("unexpected end of input")
    tok = tokens[pos]
    if tok == "(":
        form = []
        pos += 1
        while True:
            if pos >= len(tokens):
                raise SyntaxError("missing closing parenthesis")
            if tokens[pos] == ")":
                return form, pos + 1
            sub, pos = _read(tokens, pos)
            form.append(sub)
    if tok == ")":
        raise SyntaxError("unexpected )")
    return _atom(tok), pos + 1


def parse(text: str) -> list:
    """Parse source text into a list of top-level forms."""
    tokens = tokenize(text)
    forms = []
    pos = 0
    while pos < len(tokens):
        form, pos = _read(tokens, pos)
        forms.append(form)
    return forms


def parse_one(text: str):
    """Parse source text that contains exactly one top-level form."""
    forms = parse(text)
    if len(forms) != 1:
        raise SyntaxError(f"expected exactly one form, got {len(forms)}")
    return forms[0]


def to_string(form) -> str:
    """Render a form back to (approximate) source text."""
    if isinstance(form, bool):
        return "true" if form else "false"
    if form is None:
        return "nil"
    if isinstance(form, Symbol):
        return str(form)
    if isinstance(form, str):
        return '"' + form + '"'
    if isinstance(form, list):
        return "(" + " ".join(to_string(f) for f in form) + ")"
    return repr(form)
