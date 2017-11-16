from parcels.field import Field
from parcels.loggers import logger
import ast
import cgen as c
from collections import OrderedDict
import math
import random


class IntrinsicNode(ast.AST):
    def __init__(self, obj, ccode):
        self.obj = obj
        self.ccode = ccode


class FieldSetNode(IntrinsicNode):
    def __getattr__(self, attr):
        if isinstance(getattr(self.obj, attr), Field):
            return FieldNode(getattr(self.obj, attr),
                             ccode="%s->%s" % (self.ccode, attr))
        else:
            return ConstNode(getattr(self.obj, attr),
                             ccode="%s" % (attr))


class FieldNode(IntrinsicNode):
    def __getitem__(self, attr):
        return FieldEvalNode(self.obj, attr)


class FieldEvalNode(IntrinsicNode):
    def __init__(self, field, args, var):
        self.field = field
        self.args = args
        self.var = var


class ConstNode(IntrinsicNode):
    def __getitem__(self, attr):
        return attr


class MathNode(IntrinsicNode):
    symbol_map = {'pi': 'M_PI', 'e': 'M_E'}

    def __getattr__(self, attr):
        if hasattr(math, attr):
            if attr in self.symbol_map:
                attr = self.symbol_map[attr]
            return IntrinsicNode(None, ccode=attr)
        else:
            raise AttributeError("""Unknown math function encountered: %s"""
                                 % attr)


class RandomNode(IntrinsicNode):
    symbol_map = {'random': 'parcels_random',
                  'uniform': 'parcels_uniform',
                  'randint': 'parcels_randint',
                  'normalvariate': 'parcels_normalvariate',
                  'seed': 'parcels_seed'}

    def __getattr__(self, attr):
        if hasattr(random, attr):
            if attr in self.symbol_map:
                attr = self.symbol_map[attr]
            return IntrinsicNode(None, ccode=attr)
        else:
            raise AttributeError("""Unknown random function encountered: %s"""
                                 % attr)


class ErrorCodeNode(IntrinsicNode):
    symbol_map = {'Success': 'SUCCESS', 'Repeat': 'REPEAT', 'Delete': 'DELETE',
                  'Error': 'ERROR', 'ErrorOutOfBounds': 'ERROR_OUT_OF_BOUNDS'}

    def __getattr__(self, attr):
        if attr in self.symbol_map:
            attr = self.symbol_map[attr]
            return IntrinsicNode(None, ccode=attr)
        else:
            raise AttributeError("""Unknown math function encountered: %s"""
                                 % attr)


class ParticleAttributeNode(IntrinsicNode):
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        self.ccode = "%s->%s" % (obj.ccode, attr)
        self.ccode_index_var = None

        if self.attr == 'lon':
            self.ccode_index_var = "%s->%s" % (self.obj.ccode, "xi")
        elif self.attr == 'lat':
            self.ccode_index_var = "%s->%s" % (self.obj.ccode, "yi")

    @property
    def pyast_index_update(self):
        pyast = ast.Assign()
        pyast.targets = [IntrinsicNode(None, ccode='err')]
        pyast.value = IntrinsicNode(None, ccode=self.ccode_index_update)
        return pyast

    @property
    def ccode_index_update(self):
        """C-code for the index update requires after updating p.lon/p.lat"""
        if self.attr == 'lon':
            return "search_linear_float(%s, U->xdim, U->lon, &(%s)); CHECKERROR(err)" \
                % (self.ccode, self.ccode_index_var)
        if self.attr == 'lat':
            return "search_linear_float(%s, U->ydim, U->lat, &(%s)); CHECKERROR(err)" \
                % (self.ccode, self.ccode_index_var)
        if self.attr == 'depth':
            return "search_linear_float(%s, U->zdim, U->depth, &(%s)); CHECKERROR(err)" \
                % (self.ccode, self.ccode_index_var)
        return ""


class ParticleNode(IntrinsicNode):
    def __getattr__(self, attr):
        if attr in [v.name for v in self.obj.variables]:
            return ParticleAttributeNode(self, attr)
        elif attr in ['delete']:
            return ParticleAttributeNode(self, 'state')
        else:
            raise AttributeError("""Particle type %s does not define attribute "%s".
Please add '%s' to %s.users_vars or define an appropriate sub-class."""
                                 % (self.obj, attr, attr, self.obj))


class IntrinsicTransformer(ast.NodeTransformer):
    """AST transformer that catches any mention of intrinsic variable
    names, such as 'particle' or 'fieldset', inserts placeholder objects
    and propagates attribute access."""

    def __init__(self, fieldset, ptype):
        self.fieldset = fieldset
        self.ptype = ptype

        # Counter and variable names for temporaries
        self._tmp_counter = 0
        self.tmp_vars = []
        # A stack of additonal staements to be inserted
        self.stmt_stack = []

    def get_tmp(self):
        """Create a new temporary veriable name"""
        tmp = "tmp%d" % self._tmp_counter
        self._tmp_counter += 1
        self.tmp_vars += [tmp]
        return tmp

    def visit_Name(self, node):
        """Inject IntrinsicNode objects into the tree according to keyword"""
        if node.id == 'fieldset':
            node = FieldSetNode(self.fieldset, ccode='fset')
        elif node.id == 'particle':
            node = ParticleNode(self.ptype, ccode='particle')
        elif node.id in ['ErrorCode', 'Error']:
            node = ErrorCodeNode(math, ccode='')
        elif node.id == 'math':
            node = MathNode(math, ccode='')
        elif node.id == 'random':
            node = RandomNode(math, ccode='')
        return node

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        if isinstance(node.value, IntrinsicNode):
            return getattr(node.value, node.attr)
        else:
            raise NotImplementedError("Cannot propagate attribute access to C-code")

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        # If we encounter field evaluation we replace it with a
        # temporary variable and put the evaluation call on the stack.
        if isinstance(node.value, FieldNode):
            tmp = self.get_tmp()
            # Insert placeholder node for field eval ...
            self.stmt_stack += [FieldEvalNode(node.value, node.slice, tmp)]
            # .. and return the name of the temporary that will be populated
            return ast.Name(id=tmp)
        elif isinstance(node.value, IntrinsicNode):
            raise NotImplementedError("Subscript not implemented for object type %s"
                                      % type(node.value).__name__)
        else:
            return node

    def visit_AugAssign(self, node):
        node.target = self.visit(node.target)
        node.op = self.visit(node.op)
        node.value = self.visit(node.value)
        stmts = [node]

        # Capture p.lat/p.lon updates and insert p.xi/p.yi updates
        if isinstance(node.target, ParticleAttributeNode) \
           and node.target.ccode_index_var is not None:
            stmts += [node.target.pyast_index_update]

        # Inject statements from the stack
        if len(self.stmt_stack) > 0:
            stmts = self.stmt_stack + stmts
            self.stmt_stack = []
        return stmts

    def visit_Assign(self, node):
        node.targets = [self.visit(t) for t in node.targets]
        node.value = self.visit(node.value)
        stmts = [node]

        # Capture p.lat/p.lon updates and insert p.xi/p.yi updates
        if isinstance(node.targets[0], ParticleAttributeNode) \
           and node.targets[0].ccode_index_var is not None:
            stmts += [node.targets[0].pyast_index_update]

        # Inject statements from the stack
        if len(self.stmt_stack) > 0:
            stmts = self.stmt_stack + stmts
            self.stmt_stack = []
        return stmts

    def visit_Call(self, node):
        node.func = self.visit(node.func)
        node.args = [self.visit(a) for a in node.args]
        if isinstance(node.func, ParticleAttributeNode) \
           and node.func.attr == 'state':
            node = IntrinsicNode(node, "return DELETE")
        return node


class TupleSplitter(ast.NodeTransformer):
    """AST transformer that detects and splits Pythonic tuple
    assignments into multiple statements for conversion to C."""

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) \
           and isinstance(node.value, ast.Tuple):
            t_elts = node.targets[0].elts
            v_elts = node.value.elts
            if len(t_elts) != len(v_elts):
                raise AttributeError("Tuple lenghts in assignment do not agree")
            node = [ast.Assign() for _ in t_elts]
            for n, t, v in zip(node, t_elts, v_elts):
                n.targets = [t]
                n.value = v
        return node


class KernelGenerator(ast.NodeVisitor):
    """Code generator class that translates simple Python kernel
    functions into C functions by populating and accessing the `ccode`
    attriibute on nodes in the Python AST."""

    # Intrinsic variables that appear as function arguments
    kernel_vars = ['particle', 'fieldset', 'time', 'dt', 'output_time', 'tol']
    array_vars = []

    def __init__(self, fieldset, ptype):
        self.fieldset = fieldset
        self.ptype = ptype
        self.field_args = OrderedDict()
        # Hack alert: JIT requires U field to update fieldset indexes
        self.field_args['U'] = fieldset.U
        self.const_args = OrderedDict()

    def generate(self, py_ast, funcvars):
        # Untangle Pythonic tuple-assignment statements
        py_ast = TupleSplitter().visit(py_ast)

        # Replace occurences of intrinsic objects in Python AST
        transformer = IntrinsicTransformer(self.fieldset, self.ptype)
        py_ast = transformer.visit(py_ast)

        # Generate C-code for all nodes in the Python AST
        self.visit(py_ast)
        self.ccode = py_ast.ccode

        # Insert variable declarations for non-instrinsics
        # Make sure that repeated variables are not declared more than
        # once. If variables occur in multiple Kernels, give a warning
        used_vars = []
        for kvar in funcvars:
            if kvar in used_vars:
                logger.warning(kvar+" declared in multiple Kernels")
                funcvars.remove(kvar)
            else:
                used_vars.append(kvar)
        for kvar in self.kernel_vars + self.array_vars:
            if kvar in funcvars:
                funcvars.remove(kvar)
        self.ccode.body.insert(0, c.Value('ErrorCode', 'err'))
        if len(funcvars) > 0:
            self.ccode.body.insert(0, c.Value("float", ", ".join(funcvars)))
        if len(transformer.tmp_vars) > 0:
            self.ccode.body.insert(0, c.Value("float", ", ".join(transformer.tmp_vars)))

        return self.ccode

    def visit_FunctionDef(self, node):
        # Generate "ccode" attribute by traversing the Python AST
        for stmt in node.body:
            if not (hasattr(stmt, 'value') and type(stmt.value) is ast.Str):  # ignore docstrings
                self.visit(stmt)

        # Create function declaration and argument list
        decl = c.Static(c.DeclSpecifier(c.Value("ErrorCode", node.name), spec='inline'))
        args = [c.Pointer(c.Value(self.ptype.name, "particle")),
                c.Value("double", "time"), c.Value("float", "dt")]
        for field, _ in self.field_args.items():
            args += [c.Pointer(c.Value("CField", "%s" % field))]
        for const, _ in self.const_args.items():
            args += [c.Value("float", const)]

        # Create function body as C-code object
        body = [stmt.ccode for stmt in node.body if not (hasattr(stmt, 'value') and type(stmt.value) is ast.Str)]
        body += [c.Statement("return SUCCESS")]
        node.ccode = c.FunctionBody(c.FunctionDeclaration(decl, args), c.Block(body))

    def visit_Call(self, node):
        """Generate C code for simple C-style function calls. Please
        note that starred and keyword arguments are currently not
        supported."""
        for a in node.args:
            self.visit(a)
        ccode_args = ", ".join([a.ccode for a in node.args])
        try:
            node.ccode = "%s(%s)" % (node.func.ccode, ccode_args)
        except:
            raise RuntimeError("Error in converting Kernel to C. See http://oceanparcels.org/#writing-parcels-kernels for hints and tips")

    def visit_Name(self, node):
        """Catches any mention of intrinsic variable names, such as
        'particle' or 'fieldset' and inserts our placeholder objects"""
        if node.id == 'True':
            node.id = "1"
        if node.id == 'False':
            node.id = "0"
        node.ccode = node.id

    def visit_NameConstant(self, node):
        if node.value is True:
            node.ccode = "1"
        if node.value is False:
            node.ccode = "0"

    def visit_Expr(self, node):
        self.visit(node.value)
        node.ccode = c.Statement(node.value.ccode)

    def visit_Assign(self, node):
        self.visit(node.targets[0])
        self.visit(node.value)
        if isinstance(node.value, ast.List):
            # Detect in-place initialisation of multi-dimensional arrays
            tmp_node = node.value
            decl = c.Value('float', node.targets[0].id)
            while isinstance(tmp_node, ast.List):
                decl = c.ArrayOf(decl, len(tmp_node.elts))
                if isinstance(tmp_node.elts[0], ast.List):
                    # Check type and dimension are the same
                    if not all(isinstance(e, ast.List) for e in tmp_node.elts):
                        raise TypeError("Non-list element discovered in array declaration")
                    if not all(len(e.elts) == len(tmp_node.elts[0].elts) for e in tmp_node.elts):
                        raise TypeError("Irregular array length not allowed in array declaration")
                tmp_node = tmp_node.elts[0]
            node.ccode = c.Initializer(decl, node.value.ccode)
            self.array_vars += [node.targets[0].id]
        else:
            node.ccode = c.Assign(node.targets[0].ccode, node.value.ccode)

    def visit_AugAssign(self, node):
        self.visit(node.target)
        self.visit(node.op)
        self.visit(node.value)
        node.ccode = c.Statement("%s %s= %s" % (node.target.ccode,
                                                node.op.ccode,
                                                node.value.ccode))

    def visit_If(self, node):
        self.visit(node.test)
        for b in node.body:
            self.visit(b)
        for b in node.orelse:
            self.visit(b)
        body = c.Block([b.ccode for b in node.body])
        orelse = c.Block([b.ccode for b in node.orelse]) if len(node.orelse) > 0 else None
        node.ccode = c.If(node.test.ccode, body, orelse)

    def visit_Compare(self, node):
        self.visit(node.left)
        assert(len(node.ops) == 1)
        self.visit(node.ops[0])
        assert(len(node.comparators) == 1)
        self.visit(node.comparators[0])
        node.ccode = "%s %s %s" % (node.left.ccode, node.ops[0].ccode,
                                   node.comparators[0].ccode)

    def visit_Index(self, node):
        self.visit(node.value)
        node.ccode = node.value.ccode

    def visit_Tuple(self, node):
        for e in node.elts:
            self.visit(e)
        node.ccode = tuple([e.ccode for e in node.elts])

    def visit_List(self, node):
        for e in node.elts:
            self.visit(e)
        node.ccode = "{" + ", ".join([e.ccode for e in node.elts]) + "}"

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)
        if isinstance(node.value, FieldNode):
            node.ccode = node.value.__getitem__(node.slice.ccode).ccode
        elif isinstance(node.value, IntrinsicNode):
            raise NotImplementedError("Subscript not implemented for object type %s"
                                      % type(node.value).__name__)
        else:
            node.ccode = "%s[%s]" % (node.value.ccode, node.slice.ccode)

    def visit_UnaryOp(self, node):
        self.visit(node.op)
        self.visit(node.operand)
        node.ccode = "%s(%s)" % (node.op.ccode, node.operand.ccode)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.op)
        self.visit(node.right)
        node.ccode = "(%s %s %s)" % (node.left.ccode, node.op.ccode, node.right.ccode)
        node.s_print = True

    def visit_Add(self, node):
        node.ccode = "+"

    def visit_UAdd(self, node):
        node.ccode = "+"

    def visit_Sub(self, node):
        node.ccode = "-"

    def visit_USub(self, node):
        node.ccode = "-"

    def visit_Mult(self, node):
        node.ccode = "*"

    def visit_Div(self, node):
        node.ccode = "/"

    def visit_Mod(self, node):
        node.ccode = "%"

    def visit_Num(self, node):
        node.ccode = str(node.n)

    def visit_BoolOp(self, node):
        self.visit(node.op)
        for v in node.values:
            self.visit(v)
        op_str = " %s " % node.op.ccode
        node.ccode = op_str.join([v.ccode for v in node.values])

    def visit_Eq(self, node):
        node.ccode = "=="

    def visit_Lt(self, node):
        node.ccode = "<"

    def visit_LtE(self, node):
        node.ccode = "<="

    def visit_Gt(self, node):
        node.ccode = ">"

    def visit_GtE(self, node):
        node.ccode = ">="

    def visit_And(self, node):
        node.ccode = "&&"

    def visit_Or(self, node):
        node.ccode = "||"

    def visit_Not(self, node):
        node.ccode = "!"

    def visit_While(self, node):
        self.visit(node.test)
        for b in node.body:
            self.visit(b)
        if len(node.orelse) > 0:
            raise RuntimeError("Else clause in while clauses cannot be translated to C")
        body = c.Block([b.ccode for b in node.body])
        node.ccode = c.DoWhile(node.test.ccode, body)

    def visit_For(self, node):
        raise RuntimeError("For loops cannot be translated to C")

    def visit_Break(self, node):
        node.ccode = c.Statement("break")

    def visit_FieldNode(self, node):
        """Record intrinsic fields used in kernel"""
        self.field_args[node.obj.name] = node.obj

    def visit_ConstNode(self, node):
        self.const_args[node.ccode] = node.obj

    def visit_FieldEvalNode(self, node):
        self.visit(node.field)
        self.visit(node.args)
        ccode_eval = node.field.obj.ccode_eval(node.var, *node.args.ccode)
        ccode_conv = node.field.obj.ccode_convert(*node.args.ccode)
        node.ccode = c.Block([c.Assign("err", ccode_eval),
                              c.Statement("%s *= %s" % (node.var, ccode_conv)),
                              c.Statement("CHECKERROR(err)")])

    def visit_Return(self, node):
        self.visit(node.value)
        node.ccode = c.Statement('return %s' % node.value.ccode)

    def visit_Print(self, node):
        for n in node.values:
            self.visit(n)
        if hasattr(node.values[0], 's'):
            node.ccode = c.Statement('printf("%s\\n")' % (n.ccode))
            return
        if hasattr(node.values[0], 's_print'):
            args = node.values[0].right.ccode
            s = ('printf("%s\\n"' % node.values[0].left.ccode)
            if isinstance(args, str):
                s = s + (", %s)" % args)
            else:
                for arg in args:
                    s = s + (", %s" % arg)
                s = s + ")"
            node.ccode = c.Statement(s)
            return
        vars = ', '.join([n.ccode for n in node.values])
        int_vars = ['particle->id', 'particle->xi', 'particle->yi', 'particle->zi']
        stat = ', '.join(["%d" if n.ccode in int_vars else "%f" for n in node.values])
        node.ccode = c.Statement('printf("%s\\n", %s)' % (stat, vars))

    def visit_Str(self, node):
        node.ccode = node.s


class LoopGenerator(object):
    """Code generator class that adds type definitions and the outer
    loop around kernel functions to generate compilable C code."""

    def __init__(self, fieldset, ptype=None):
        self.fieldset = fieldset
        self.ptype = ptype

    def generate(self, funcname, field_args, const_args, kernel_ast):
        ccode = []

        # Add include for Parcels and math header
        ccode += [str(c.Include("parcels.h", system=False))]
        ccode += [str(c.Include("math.h", system=False))]

        # Generate type definition for particle type
        vdecl = [c.POD(v.dtype, v.name) for v in self.ptype.variables]
        ccode += [str(c.Typedef(c.GenerableStruct("", vdecl, declname=self.ptype.name)))]

        # Insert kernel code
        ccode += [str(kernel_ast)]

        # Generate outer loop for repeated kernel invocation
        args = [c.Value("int", "num_particles"),
                c.Pointer(c.Value(self.ptype.name, "particles")),
                c.Value("double", "endtime"), c.Value("float", "dt")]
        for field, _ in field_args.items():
            args += [c.Pointer(c.Value("CField", "%s" % field))]
        for const, _ in const_args.items():
            args += [c.Value("float", const)]
        fargs_str = ", ".join(['particles[p].time', 'sign * __dt'] + list(field_args.keys()) + list(const_args.keys()))
        # Inner loop nest for forward runs
        sign = c.Assign("sign", "dt > 0. ? 1. : -1.")
        dt_pos = c.Assign("__dt", "fmin(fabs(particles[p].dt), fabs(endtime - particles[p].time))")
        body = [c.Assign("res", "%s(&(particles[p]), %s)" % (funcname, fargs_str))]
        body += [c.Assign("particles[p].state", "res")]  # Store return code on particle
        body += [c.If("res == SUCCESS", c.Block([c.Statement("particles[p].time += sign * __dt"),
                                                 dt_pos, c.Statement("continue")]))]
        body += [c.If("res == REPEAT", c.Block([dt_pos, c.Statement("continue")]),
                      c.Statement("break"))]

        time_loop = c.While("__dt > __tol", c.Block(body))
        part_loop = c.For("p = 0", "p < num_particles", "++p", c.Block([dt_pos, time_loop]))
        fbody = c.Block([c.Value("int", "p"), c.Value("ErrorCode", "res"),
                         c.Value("double", "__dt, __tol, sign"), c.Assign("__tol", "1.e-6"),
                         sign, part_loop])
        fdecl = c.FunctionDeclaration(c.Value("void", "particle_loop"), args)
        ccode += [str(c.FunctionBody(fdecl, fbody))]
        return "\n\n".join(ccode)
