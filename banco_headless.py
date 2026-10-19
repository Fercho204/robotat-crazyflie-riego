"""Utilidades offline: extrae definiciones AST sin importar módulos de hardware."""
import ast
from pathlib import Path
import numpy as np


def extraer(archivo, nombres, entorno=None):
    tree = ast.parse(Path(archivo).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in nombres]
    scope = {'np': np, '__name__': 'offline'}
    scope.update(entorno or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), archivo, 'exec'), scope)
    return scope


def extraer_metodo(archivo, clase, metodo, entorno):
    tree = ast.parse(Path(archivo).read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clase)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == metodo)
    scope = dict(entorno)
    exec(compile(ast.Module(body=[node], type_ignores=[]), archivo, 'exec'), scope)
    return scope[metodo]


class Banco:
    def __init__(self):
        self.aciertos = self.fallos = 0

    def comprobar(self, nombre, ok, antes, despues):
        self.aciertos += int(bool(ok))
        self.fallos += int(not ok)
        print(f'{nombre}: antes={antes} despues={despues} ' + ('OK' if ok else 'FALLO'))

    def terminar(self):
        print(f'FINAL: {self.aciertos} aciertos / {self.fallos} fallos')
        return int(self.fallos > 0)
