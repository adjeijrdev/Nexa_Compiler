#!/usr/bin/env python3
# Nexa compiler: integers, strings, chars; LLVM JIT with llvmlite
import re
import sys
import os
import ctypes
from dataclasses import dataclass
from typing import List, Optional, Dict

from llvmlite import ir, binding as llvm

# -----------------------------
# 1) Lexer  [ AMINA ]
# -----------------------------
TOKEN_SPEC = [
    ("LET",      r"let\b"),
    ("PRINT",    r"print\b"),
    ("STRING",   r'"([^"\\]|\\.)*"'),     # "Hello\n"
    ("CHAR",     r"'([^'\\]|\\.)'"),      # 'A' or '\n'
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
    ("NUMBER",   r"\d+"),
    ("PLUS",     r"\+"),
    ("MINUS",    r"-"),
    ("COMMENT",  r"\#.*"),
    ("STAR",     r"\*"),
    ("SLASH",    r"/"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("ASSIGN",   r"="),
    ("SEMI",     r";"),
    ("SKIP",     r"[ \t\r\n]+"),
    ("MISMATCH", r"."),
]
TOK_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))

@dataclass
class Token:
    type: str
    value: str
    pos: int

def _unescape(s: str) -> str:
    escapes = {
        r"\\": "\\",
        r"\"": "\"",
        r"\'": "'",
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\0": "\0",
    }
    out = ""
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            pair = s[i:i+2]
            if pair in escapes:
                out += escapes[pair]; i += 2; continue
        out += s[i]; i += 1
    return out

def lex(src: str) -> List[Token]:
    tokens: List[Token] = []
    for m in TOK_REGEX.finditer(src):
        kind = m.lastgroup
        value = m.group()
        if kind == "SKIP" or kind == "COMMENT":
            continue
        if kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value!r} at {m.start()}")
        tokens.append(Token(kind, value, m.start()))
    return tokens

# --------------------------------------
# 2) AST nodes  [ COURAGE ]
# --------------------------------------
class Stmt: ...
class Expr: ...

@dataclass
class Program:
    body: List[Stmt]

@dataclass
class LetStmt(Stmt):
    name: str
    expr: Expr

@dataclass
class PrintStmt(Stmt):
    expr: Expr

@dataclass
class Number(Expr):
    value: int

@dataclass
class StringLit(Expr):
    value: str  # unescaped Python string

@dataclass
class CharLit(Expr):
    value: str  # single-character Python string

@dataclass
class Var(Expr):
    name: str

@dataclass
class BinOp(Expr):
    op: str   # '+','-','*','/'
    left: Expr
    right: Expr

# ------------------------------------------
# 3) Parser (recursive descent)  [ JOHN]
# ------------------------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Optional[Token]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, kind: str) -> Token:
        tok = self._peek()
        if not tok or tok.type != kind:
            raise SyntaxError(f"Expected {kind} at pos {tok.pos if tok else 'EOF'}")
        self.i += 1
        return tok

    def _match(self, *kinds) -> Optional[Token]:
        tok = self._peek()
        if tok and tok.type in kinds:
            self.i += 1
            return tok
        return None

    def parse(self) -> Program:
        body: List[Stmt] = []
        while self._peek() is not None:
            body.append(self.statement())
        return Program(body)

    def statement(self) -> Stmt:
        if self._match("LET"):
            ident = self._eat("IDENT").value
            self._eat("ASSIGN")
            e = self.expr()
            self._eat("SEMI")
            return LetStmt(ident, e)
        elif self._match("PRINT"):
            e = self.expr()
            self._eat("SEMI")
            return PrintStmt(e)
        else:
            tok = self._peek()
            raise SyntaxError(f"Unexpected token {tok.type if tok else 'EOF'} at {tok.pos if tok else 'EOF'}")

    def expr(self) -> Expr:
        node = self.term()
        while True:
            tok = self._match("PLUS", "MINUS")
            if not tok: break
            right = self.term()
            node = BinOp("+" if tok.type=="PLUS" else "-", node, right)
        return node

    def term(self) -> Expr:
        node = self.factor()
        while True:
            tok = self._match("STAR", "SLASH")
            if not tok: break
            right = self.factor()
            node = BinOp("*" if tok.type=="STAR" else "/", node, right)
        return node

    def factor(self) -> Expr:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected EOF in factor")

        if self._match("NUMBER"):
            return Number(int(tok.value))
        if self._match("STRING"):
            raw = tok.value[1:-1]
            return StringLit(_unescape(raw))
        if self._match("CHAR"):
            raw = tok.value[1:-1]
            ch = _unescape(raw)
            if len(ch) != 1:
                raise SyntaxError(f"Char literal must be 1 char at pos {tok.pos}")
            return CharLit(ch)
        if self._match("IDENT"):
            return Var(tok.value)
        if self._match("LPAREN"):
            node = self.expr()
            self._eat("RPAREN")
            return node
        raise SyntaxError(f"Bad factor at pos {tok.pos}")

# ---------------------------------------------------------
# 4) Codegen to LLVM IR (llvmlite)  [ KENNETH ]
# ---------------------------------------------------------
class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="nexa")
        self.i64 = ir.IntType(64)
        self.i32 = ir.IntType(32)
        self.i8  = ir.IntType(8)
        self.i8p = self.i8.as_pointer()

        # extern void print_int(i64)
        self.print_fn_ty = ir.FunctionType(ir.VoidType(), [self.i64])
        self.print_fn = ir.Function(self.module, self.print_fn_ty, name="print_int")

        # extern i32 puts(i8*)
        self.puts_ty = ir.FunctionType(self.i32, [self.i8p])
        self.puts_fn = ir.Function(self.module, self.puts_ty, name="puts")

        # extern i32 putchar(i32)
        self.putchar_ty = ir.FunctionType(self.i32, [self.i32])
        self.putchar_fn = ir.Function(self.module, self.putchar_ty, name="putchar")

        # define i32 @main()
        main_ty = ir.FunctionType(self.i32, [])
        self.main_fn = ir.Function(self.module, main_ty, name="main")
        block = self.main_fn.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        self.vars: Dict[str, ir.AllocaInstr] = {}
        self._str_counter = 0  # for unique global names

    # create a global null-terminated string and return i8* to its first char
    def cstr_ptr(self, text: str) -> ir.Value:
        data = text.encode("utf-8") + b"\x00"
        n = len(data)
        arr_ty = ir.ArrayType(self.i8, n)
        name = f".str.{self._str_counter}"
        self._str_counter += 1

        gv = ir.GlobalVariable(self.module, arr_ty, name)
        gv.linkage = 'internal'
        gv.global_constant = True
        gv.initializer = ir.Constant(arr_ty, bytearray(data))

        zero = ir.Constant(self.i32, 0)
        ptr = self.builder.gep(gv, [zero, zero], inbounds=True)  # i8* to first element
        return ptr

    def codegen(self, prog: Program) -> ir.Module:
        for stmt in prog.body:
            self.codegen_stmt(stmt)
        self.builder.ret(self.i32(0))
        return self.module

    def alloca(self, name: str):
        ptr = self.builder.alloca(self.i64, name=name)
        self.vars[name] = ptr
        return ptr

    def codegen_stmt(self, stmt: Stmt):
        if isinstance(stmt, LetStmt):
            val = self.codegen_expr(stmt.expr)
            ptr = self.vars.get(stmt.name) or self.alloca(stmt.name)
            self.builder.store(val, ptr)
        elif isinstance(stmt, PrintStmt):
            if isinstance(stmt.expr, StringLit):
                p = self.codegen_expr(stmt.expr)  # i8*
                self.builder.call(self.puts_fn, [p])
            elif isinstance(stmt.expr, CharLit):
                ch_val = self.codegen_expr(stmt.expr)  # i64
                ch_i32 = self.builder.trunc(ch_val, self.i32) if isinstance(ch_val.type, ir.IntType) and ch_val.type.width > 32 else ch_val
                self.builder.call(self.putchar_fn, [ch_i32])
                self.builder.call(self.putchar_fn, [ir.Constant(self.i32, ord('\n'))])
            else:
                val = self.codegen_expr(stmt.expr)   # i64
                self.builder.call(self.print_fn, [val])
        else:
            raise ValueError("Unknown statement")

    def codegen_expr(self, expr: Expr) -> ir.Value:
        if isinstance(expr, Number):
            return ir.Constant(self.i64, expr.value)
        if isinstance(expr, StringLit):
            return self.cstr_ptr(expr.value)  # i8*
        if isinstance(expr, CharLit):
            return ir.Constant(self.i64, ord(expr.value))
        if isinstance(expr, Var):
            ptr = self.vars.get(expr.name)
            if ptr is None:
                raise NameError(f"Undefined variable '{expr.name}'")
            return self.builder.load(ptr, name=expr.name)
        if isinstance(expr, BinOp):
            l = self.codegen_expr(expr.left)
            r = self.codegen_expr(expr.right)
            if expr.op == "+": return self.builder.add(l, r)
            if expr.op == "-": return self.builder.sub(l, r)
            if expr.op == "*": return self.builder.mul(l, r)
            if expr.op == "/": return self.builder.sdiv(l, r)
        raise ValueError("Unknown expression")

# -----------------------------
# 5) JIT driver    [ DANIEL]
# -----------------------------
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

PRINTED: List[int] = []

@ctypes.CFUNCTYPE(None, ctypes.c_longlong)
def py_print_int(n):
    PRINTED.append(int(n))
    print(n)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
def py_puts(cstr):
    try:
        s = ctypes.cast(cstr, ctypes.c_char_p).value
        print((s or b"").decode("utf-8"))
        return 0
    except Exception:
        return -1

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
def py_putchar(ch):
    try:
        sys.stdout.write(chr(ch & 0xFF))
        sys.stdout.flush()
        return ch & 0xFF
    except Exception:
        return -1

def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine

def compile_ir(engine, llvm_ir: str):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod

def run(source_code: str):
    tokens = lex(source_code)
    prog = Parser(tokens).parse()
    cg = CodeGen()
    module = cg.codegen(prog)

    # Bind runtime symbols
    llvm.add_symbol("print_int", ctypes.cast(py_print_int, ctypes.c_void_p).value)
    llvm.add_symbol("puts", ctypes.cast(py_puts, ctypes.c_void_p).value)
    llvm.add_symbol("putchar", ctypes.cast(py_putchar, ctypes.c_void_p).value)

    engine = create_execution_engine()
    compile_ir(engine, str(module))

    main_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(main_ptr)
    return cfunc()

# -------------------------------------------------
# 6) CLI (accept only .nx / .nexa)  [ DANIEL ]
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: nexa <sourcefile.nx | sourcefile.nexa>")
        sys.exit(1)

    filename = sys.argv[1]
    if not (filename.endswith(".nx") or filename.endswith(".nexa")):
        print("Error: Nexa source files must have .nx or .nexa extension")
        sys.exit(1)

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

    with open(filename, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        exit_code = run(source_code)
        sys.exit(exit_code)
    except Exception as e:
        print(f"Compiler error: {e}")
        sys.exit(2)
