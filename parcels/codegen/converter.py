from parcels.codegen import ir
import ast
import cgen as c


__all__ = ['IRConverter', 'CodeGenerator']

op2str = {
    ast.Add: '+', ast.Sub: '-',
    ast.Mult: '*', ast.Div: '/'
}


class IRConverter(ast.NodeTransformer):
    """Converter from Python AST into Parcels IR.

    """

    def __init__(self, grid, ptype):
        self.grid = grid
        self.ptype = ptype

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            assert(arg.id in ['particle', 'grid', 'time', 'dt'])

        ir_nodes = [self.visit(n) for n in node.body]
        return ir.Root(name=node.name, nodes=ir_nodes)

    def visit_Assign(self, node):
        # TODO: Deal with tuple assignments
        target = self.visit(node.targets[0])
        expr = self.visit(node.value)

        # Add statement to code block
        return ir.Assign(target, expr, op='=')

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        expr = self.visit(node.value)

        # Add statement to code block
        return ir.Assign(target, expr, op='%s=' % op2str[type(node.op)])

    def visit_Name(self, node):
        if node.id == 'grid':
            return ir.GridIntrinsic(self.grid)
        elif node.id == 'particle':
            return ir.ParticleIntrinsic(self.ptype)
        elif node.id in ['time', 'dt']:
            return ir.Variable(node.id, declared=True)
        else:
            return ir.Variable(node.id, declared=False)

    def visit_Num(self, node):
        return ir.Constant(node.n)

    def visit_BinOp(self, node):
        return ir.BinaryOperator(expr1=self.visit(node.left),
                                 expr2=self.visit(node.right),
                                 op=op2str[type(node.op)])

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        return getattr(obj, node.attr)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Tuple(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Subscript(self, node):
        field = self.visit(node.value)
        assert(isinstance(field, ir.FieldIntrinsic))
        args = self.visit(node.slice)
        return ir.FieldEvalIntrinsic(field.field, args)


class VariableFinder(ast.NodeVisitor):
    """Utitility visitor to identify local variables for declaration

    """

    def __init__(self, ast):
        self._undeclared = set()
        self._fields = set()

        self.visit(ast)

    @property
    def fields(self):
        return sorted(list(self._fields))

    @property
    def undeclared(self):
        return sorted(list(self._undeclared))

    def visit_Variable(self, node):
        if not node.declared:
            self._undeclared.update([node.name])

    def visit_FieldEvalIntrinsic(self, node):
        self._fields.update([node.field.name])


class HoistFieldEvaluation(ast.NodeTransformer):
    """Hoists field evaluation to the root level and assign
    temporaries instead.

    """

    def __init__(self, ):
        self.statements = []
        self._tmp_counter = 0

    def get_temporary(self):
        """Create a new temporary variable name"""
        tmp = "tmp%d" % self._tmp_counter
        self._tmp_counter += 1
        return tmp

    def visit_Root(self, node):
        name = node.name
        for child in node.children:
            # Build a global list of root statements
            # that the field eval method can add to
            self.statements += [self.visit(child)]
        return ir.Root(name=name, nodes=self.statements)

    def visit_FieldEvalIntrinsic(self, node):
        """Insert temporary and add eval call to root statements"""
        variable = ir.Variable(name=self.get_temporary())
        node.var = variable
        self.statements += [node]
        return variable


class CodeGenerator(ast.NodeVisitor):
    """IR visitor that generates the final C code.

    """

    def generate(self, ast):
        assert(isinstance(ast, ir.Root))
        return self.visit(ast)

    def generic_visit(self, node):
        body = [self.visit(child) for child in ast.iter_child_nodes(node)]
        return c.Comment("<%s: %s>" % (type(node).__name__, body))

    def visit_Root(self, node):
        return [self.visit(c) for c in node.children]

    def visit_Constant(self, node):
        return node.value

    def visit_Variable(self, node):
        return node.name

    def visit_BinaryOperator(self, node):
        return '(%s %s %s)' % (self.visit(node.children[0]), node.op,
                               self.visit(node.children[1]))

    def visit_Assign(self, node):
        target = self.visit(node.target)
        expr = self.visit(node.expr)
        return c.Statement('%s %s %s' % (target, node.op, expr))

    def visit_FieldIntrinsic(self, node):
        return 'grid->%s' % node.field.name

    def visit_FieldEvalIntrinsic(self, node):
        var = self.visit(node.var)
        args = [self.visit(arg) for arg in node.args]
        ccode_eval = node.field.ccode_eval(var, *args)
        ccode_conv = node.field.ccode_convert(*args)
        return c.Block([c.Assign("err", ccode_eval),
                        c.Statement("%s *= %s" % (var, ccode_conv)),
                        c.Statement("CHECKERROR(err)")])
