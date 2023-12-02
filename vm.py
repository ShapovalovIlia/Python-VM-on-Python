"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any], ) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.need_arg_instead_of_argval = ["binary_op", "compare_op", "kw_names_op"]
        self.kw_names: tuple[tp.Any, ...] = tuple()
        self.counter = 0
        self.index: dict[int, int] = {}
        self._TWO_PARAMS_FUNC = ["load_global", "format_value"]
        self._ONE_PARAMS_FUNC = ["push_null_op", "return_value_op", "pop_top_op",
                                 "get_iter_op", "unpack_sequence_op", "make_function_op",
                                 "binary_subscr_op", "load_assertion_error_op", "store_subscr_op",
                                 "import_star_op", "unary_positive_op", "unary_not_op", "unary_negative_op",
                                 "unary_invert_op", "delete_subscr_op"]
        self.last_exception = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def top1(self) -> tp.Any:
        return self.data_stack[-2]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        ins_lst = list(dis.get_instructions(self.code))
        for i in range(len(ins_lst)):
            self.index[ins_lst[i].offset] = i
        while True:
            if self.counter >= len(ins_lst):
                break
            func_name = ins_lst[self.counter].opname.lower()
            if func_name + "op" == "return_value_op":
                break
            if func_name in self._TWO_PARAMS_FUNC:
                getattr(self, func_name + "_op")(ins_lst[self.counter].arg, ins_lst[self.counter].argval)
            elif func_name in self.need_arg_instead_of_argval:
                getattr(self, func_name)(ins_lst[self.counter].arg)
            elif func_name + "_op" in self._ONE_PARAMS_FUNC:
                getattr(self, func_name + "_op")()
            else:
                getattr(self, func_name + "_op")(ins_lst[self.counter].argval)
            self.counter += 1

        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        args = self.popn(arg)
        kwargs: dict[str, tp.Any] = {}
        func = self.pop()
        if self.top() is None:
            self.pop()
            self.push(func(*args, **kwargs))
            return
        obj = func
        func = self.pop()
        self.push(func(obj, *args, **kwargs))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: int, argval: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        # TODO: parse all scopes
        if arg & 1 == 1:
            self.push_null_op()
        if argval in self.globals:
            self.push(self.globals[argval])
        elif argval in self.builtins:
            self.push(self.builtins[argval])
        else:
            raise NameError("global name '%s' is not defined" % arg)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        self.store_fast_op(arg)

    def store_global_op(self, name: str) -> None:
        self.globals[name] = self.pop()

    def binary_op(self, op: int) -> None:
        x, y = self.popn(2)
        self.push(BinaryOp.binary_operators_dict[op](x, y))

    def get_iter_op(self) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, jump: int) -> None:
        iter_obj = self.top()
        try:
            v = next(iter_obj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump_forward_op(jump)

    def jump_forward_op(self, arg: int) -> None:
        self.counter = self.index[arg] - 1

    def jump_backward_op(self, arg: int) -> None:
        self.counter = self.index[arg] - 1

    def unpack_sequence_op(self) -> None:
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def compare_op(self, op: int) -> None:
        x, y = self.popn(2)
        self.push(CompareOp.compare_operators_dict[op](x, y))

    def build_slice_op(self, count: int) -> None:
        if count == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        elif count == 3:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))
        else:  # pragma: no cover
            raise IndexError

    def make_function_op(self) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)

        # TODO: use arg to parse function defaults

        def f() -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount

            parsed_args: dict[str, tp.Any] = {}
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def binary_subscr_op(self) -> None:
        x, y = self.popn(2)
        self.push(operator.getitem(x, y))

    def pop_jump_forward_if_true_op(self, to: int) -> None:
        if self.pop():
            self.jump_forward_op(to)

    def pop_jump_forward_if_false_op(self, to: int) -> None:
        if not self.pop():
            self.jump_forward_op(to)

    def pop_jump_backward_if_true_op(self, to: int) -> None:
        if self.pop():
            self.jump_forward_op(to)

    def pop_jump_backward_if_false_op(self, to: int) -> None:
        if not self.pop():
            self.jump_forward_op(to)

    def load_assertion_error_op(self) -> None:
        self.push(AssertionError)

    def pop_jump_forward_if_none_op(self, to: int) -> None:
        if self.pop() is None:
            self.jump_forward_op(to)

    def build_list_op(self, count: int) -> None:
        self.push(self.popn(count))

    def store_subscr_op(self) -> None:
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def delete_subscr_op(self) -> None:
        obj, subscr = self.popn(2)
        del obj[subscr]

    def list_extend_op(self, i: int) -> None:
        seq = self.pop()
        list.extend(self.data_stack[-i], seq)

    def build_const_key_map_op(self, count: int) -> None:
        data = self.popn(count + 1)
        res = {data[-1][i]: data[i] for i in range(count)}
        self.push(res)

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def set_update_op(self, arg: int) -> None:
        seq = self.pop()
        set.update(self.data_stack[-arg], seq)

    def format_value_op(self, flags: int, useless: str) -> None:
        useless = "1" + useless * 0
        res = self.pop()
        if (flags & 3) == int(useless):
            res = str(res)
        elif (flags & 3) == 2:
            res = repr(res)
        elif (flags & 3) == 3:
            res = ascii(res)
        self.push(res)

    def build_string_op(self, count: int) -> None:
        """"
        Concatenates count strings from the stack and pushes the resulting string onto the stack.
        """
        dump = self.popn(count)
        res = ""
        for i in dump:
            res += i
        self.push(res)

    def load_method_op(self, func_name: str) -> None:
        obj = self.pop()
        self.push_null_op()
        self.push(getattr(obj, func_name))

    def unary_negative_op(self) -> None:
        x = self.pop()
        self.push(-x)

    def unary_invert_op(self) -> None:
        x = self.pop()
        self.push(~x)

    def unary_not_op(self) -> None:
        x = self.pop()
        self.push(not x)

    def load_fast_op(self, arg: str) -> None:
        self.load_name_op(arg)

    def store_fast_op(self, arg: str) -> None:
        const = self.pop()
        self.locals[arg] = const

    def list_append_op(self, i: int) -> None:
        obj = self.pop()
        self.data_stack[-i].append(obj)

    def build_map_op(self, count: int) -> None:
        obj = self.popn(count * 2)
        res = {}
        for i in range(0, len(obj), 2):
            res[obj[i]] = obj[i + 1]
        self.push(res)

    # def map_add_op(self, ):
    def nop_op(self, arg: int) -> None:
        pass

    def map_add_op(self, i: int) -> None:
        val = self.pop()
        key = self.pop()
        self.data_stack[-i][key] = val
        # dict.__setitem__(key[-i], key, val)

    def set_add_op(self, i: int) -> None:
        val = self.pop()
        self.data_stack[-i].add(val)

    def copy_op(self, i: int) -> None:
        self.push(self.data_stack[-i])

    def swap_op(self, i: int) -> None:
        self.data_stack[-i], self.data_stack[-1] = self.data_stack[-1], self.data_stack[-i]

    def is_op_op(self, invert: int) -> None:
        left, right = self.popn(2)
        if invert:
            self.push(left is not right)
        else:
            self.push(left is right)

    def store_attr_op(self, arg: str) -> None:
        obj = self.pop()
        val = self.pop()
        setattr(obj, arg, val)

    def load_attr_op(self, arg: str) -> None:
        obj = self.pop()
        self.push(getattr(obj, arg))

    def delete_attr_op(self, arg: str) -> None:
        obj = self.pop()
        delattr(obj, arg)

    def jump_if_true_or_pop_op(self, arg: int) -> None:
        if self.top():
            self.jump_forward_op(arg)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, arg: int) -> None:
        if not self.top():
            self.jump_forward_op(arg)
        else:
            self.pop()

    def unary_positive_op(self) -> None:
        x = self.pop()
        self.push(+x)

    def dict_update_op(self, i: int) -> None:
        arg = self.pop()
        dict.update(self.data_stack[-i], arg)

    def contains_op_op(self, arg: int) -> None:
        query, container = self.popn(2)
        if arg:
            self.push(query not in container)
        else:
            self.push(query in container)

    def delete_name_op(self, name: str) -> None:
        del self.locals[name]

    def delete_fast_op(self, name: str) -> None:
        self.delete_name_op(name)

    def build_tuple_op(self, count: tp.Any) -> None:
        args = self.popn(count)
        self.push(tuple(args))

    def import_name_op(self, name: str) -> None:
        level, fromlist = self.popn(2)
        self.push(
            __import__(name, self.globals, self.locals, fromlist, level)
        )

    def import_from_op(self, name: str) -> None:
        mod = self.top()
        self.push(getattr(mod, name))

    def kw_names_op(self, arg: int) -> None:
        self.kw_names = self.code.co_consts[arg]

    def import_star_op(self) -> None:
        module = self.pop()
        for attr_name in dir(module):
            if len(attr_name) > 0 and attr_name[0] == '_':
                continue
            self.locals[attr_name] = getattr(module, attr_name)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()


class CompareOp:
    EQ = 2
    NE = 3
    IS = 8
    IS_NOT = 9
    LT = 0
    LE = 1
    GT = 4
    GE = 5

    compare_operators_dict = {
        EQ: operator.eq,
        NE: operator.ne,
        IS: operator.is_,
        IS_NOT: operator.is_not,
        LT: operator.lt,
        LE: operator.le,
        GT: operator.gt,
        GE: operator.ge,
    }


class BinaryOp:
    ADD = 0
    AND = 1
    FLOOR_DIVIDE = 2
    LSHIFT = 3
    MATRIX_MULTIPLY = 4
    MULTIPLY = 5
    REMAINDER = 6
    OR = 7
    XOR = 12
    SUBTRACT = 10
    TRUE_DIVIDE = 11
    POWER = 8
    RSHIFT = 9
    INPLACE_ADD = 13
    INPLACE_SUBTRACT = 23
    INPLACE_MULTIPLY = 18
    INPLACE_TRUE_DIVIDE = 24
    INPLACE_FLOOR_DIVIDE = 15
    INPLACE_REMAINDER = 19
    INPLACE_MATRIX_MULTIPLY = 17
    INPLACE_POWER = 21
    INPLACE_LSHIFT = 16
    INPLACE_RSHIFT = 22
    INPLACE_AND = 14
    INPLACE_OR = 20
    INPLACE_XOR = 25

    binary_operators_dict = {
        ADD: operator.__add__,
        AND: operator.__and__,
        FLOOR_DIVIDE: operator.__floordiv__,
        LSHIFT: operator.__lshift__,
        MATRIX_MULTIPLY: operator.__matmul__,
        MULTIPLY: operator.__mul__,
        REMAINDER: operator.__mod__,
        OR: operator.__or__,
        XOR: operator.__xor__,
        SUBTRACT: operator.__sub__,
        TRUE_DIVIDE: operator.__truediv__,
        POWER: operator.__pow__,
        RSHIFT: operator.__rshift__,
        INPLACE_ADD: operator.__iadd__,
        INPLACE_SUBTRACT: operator.__isub__,
        INPLACE_MULTIPLY: operator.__imul__,
        INPLACE_TRUE_DIVIDE: operator.__itruediv__,
        INPLACE_FLOOR_DIVIDE: operator.__ifloordiv__,
        INPLACE_REMAINDER: operator.__imod__,
        INPLACE_MATRIX_MULTIPLY: operator.__imatmul__,
        INPLACE_POWER: operator.__ipow__,
        INPLACE_LSHIFT: operator.__ilshift__,
        INPLACE_RSHIFT: operator.__irshift__,
        INPLACE_AND: operator.__iand__,
        INPLACE_OR: operator.__ior__,
        INPLACE_XOR: operator.__ixor__,
    }
