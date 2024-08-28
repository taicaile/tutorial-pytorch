### return 语句在 with (context manager) 中

```python
class MyResource:
    def __enter__(self):
        print('Entering context.')
        return self
    def __exit__(self, *exc):
    	print('EXITING context.')
        
def fun():
    with MyResource():
        print('Returning inside with-statement.')
        return "fun() return"
    print('Returning outside with-statement.')

print(fun())

"""
当 return 在 with 语句中时，会先执行 __exit__, 然后在执行 return 语句
"""
```

### 重复使用context manager

```python
class Indenter:
    def __init__(self):
        self.level = 0

    def __enter__(self):
        self.level += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1
    
    def print(self, text):
        print('    ' * self.level + text)

with Indenter() as indent:
    indent.print('hi!')
    with indent:
        indent.print('hello')
        with indent:
            indent.print('bonjour')
    indent.print('hey')
```

### 使用context manager作为decorator

```python
from contextlib import ContextDecorator
import logging

logging.basicConfig(level=logging.INFO)

class track_entry_and_exit(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.info('Entering: %s', self.name)
    
    def __exit__(self, exc_type, exc, exc_tb):
        logging.info('Exiting: %s', self.name)

with track_entry_and_exit('widget loader'):
    print('Some time consuming activity goes here')

@track_entry_and_exit('widget loader')
def activity():
    print('Some time consuming activity goes here')
activity()
```

### 使用generator的yield语句作为decorator

```python
from contextlib import contextmanager

@contextmanager
def opening(filename):
    f = open(filename)
    try:
        print("yield f")
        yield f
    finally:
        print('f.close()')
        f.close()
        
with opening(__file__) as f:
    f.read()
    print("exit with")
```



## 问题

- 实现 `contextmanager`

- 用`yield`实现`contextmanager`

- 实现一个`contextmanager`, 既可以按照`contextmanager`用，也可以按照`decorator`功能使用

- 怎么实现 `contextmanager`功能

  ```python
  from typing import Any, Callable, TypeVar, cast
  import functools
  import inspect
  
  FuncType = Callable[..., Any]
  F = TypeVar('F', bound=FuncType)
  
  def print_fun_name(self):
      print(self.__class__.__name__, inspect.stack()[1][3])
  
  class _DecoratorContextManager():
      """Allow a context manager to be used as a decorator"""
      # call from @no_grad()
      def __call__(self, func):
          print_fun_name(self)
          # return self._wrap_generator(func)
          @functools.wraps(func)
          def decorate_context(*args, **kwargs):
              # call with no_grad()
              with self.__class__():
                  return func(*args, **kwargs)
                  
          return cast(F, decorate_context) # why need cast function
  
      def __enter__(self) -> None:
          raise NotImplementedError
  
      def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
          raise NotImplementedError
  
  class no_grad(_DecoratorContextManager):
      def __init__(self):
          self.prev = False
          print_fun_name(self)
  
      def __enter__(self):
          self.prev = True
          print_fun_name(self)
  
      def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
          print_fun_name(self)
  
  print("with no_grad()")
  with no_grad():
      print("print function")
  print("--------------------------")
  print("@no_grad")
  @no_grad()
  def print_test():
      print("print function")
  print_test()
  print("--------------------------")
  ```



