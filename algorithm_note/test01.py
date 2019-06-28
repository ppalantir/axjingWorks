class Stack(object):
        def __init__(self):
                self._elems = []

        def is_emptys(self):
                return self._elems == []

        def pushs(self, elem):
                self._elems.append(elem)

        def pops(self):
                if self.is_emptys():
                        raise ValueError
                return self._elems.pop()

        def peeks(self):
                if self.is_emptys():
                        raise ValueError
                return self._elems[-1]

def judges(expression):
        s = Stack()
        d = {'}':'{', ']':'[', ')':'('}
        for i in expression:
                if i == '[' or i == '{' or i == '(':
                        s.pushs(i)
                if i == ']' or i == '}' or i == ')':
                        
                        if s.is_emptys():
                                return 'NO'
                        
                        elif s.pops() != d[i]:
                                return 'NO'
          
        if not s.is_emptys():
                return 'NO'
        else:
                return "YES"

if __name__ == "__main__":
        t1 = r"{[()]}"
        t2 = r"{[(])}"
        t3 = r"{{[[(())]]}}"
        print("判断结果：", judges(t1))
        print("判断结果：", judges(t2))
        print("判断结果：", judges(t3))