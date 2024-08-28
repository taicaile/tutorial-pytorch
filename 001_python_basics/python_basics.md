# Python Basics for PyTorch

### `__init__` and `__call__`

```python
# class __init__ and __call__ method
t = ToTensor() # this will call __init__ to initialize class.
t() # this will call __call__ method.
```

```python
# class __call__ method
class ToTensor:
    def __init__(self, x):
        print(f"initialize ToTensor as {x}")
        self.x = x

    def __call__(self, y):
        print("ToTensor.__call__()", self.x+y)
        return "from __call__"

fs = [ToTensor(10), ToTensor(20)]

for i,f in enumerate(fs):
    print(f(i))
```

### `__repr__`

```python
# __repr__ method, represent class info
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        # return "<required parameter>"
        return super().__repr__()

required = _RequiredParameter()

print(required)
print(str(required))
```

### `__iter__` and `__next__`

```python


class DataLoader:
    '''
    implement iterator requires two method:
    1, __iter__
    2, __next__
    '''

    def __init__(self, max=100) -> None:
        self.max = max

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < self.max:
            results = 2**self.n
            self.n+=1
            return results
        else:
            raise StopIteration

for x in DataLoader(20):
    print(x)
```

### `str` vs `repr`

```python
class Student():
    def __init__(self, name) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"__repr__, name:{self.name}"

    def __str__(self) -> str:
        return f"__str__, name:{self.name}"

"""
for build-in method str() or print(), it actually invoke __str__ function.
__repr__ is called when you use repr()
"""

print("print(Student()) will call ",Student("Jane"))
print("repr(Student()) will call", repr(Student("Jona")))

# the following is common way
class Student():
    def __init__(self, name) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Student(name={self.name})"

    __str__ = __repr__
```

