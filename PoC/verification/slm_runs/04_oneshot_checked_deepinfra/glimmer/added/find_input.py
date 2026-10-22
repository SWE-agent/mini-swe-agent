import ast
src=open('agent.py').read()
tree=ast.parse(src)
lines=[n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id=='input']
print([(n.lineno, ast.unparse(n)) for n in lines])
