"""
调用栈示例
"""
def greet2(name):
    print("how are you, " + name + "?")

def bye():
    print("ok bye!")



def greet(name):
    print("hello, " + name + "!")
    greet2(name)
    print("getting ready to say bye...")
    bye()

greet("Anders")


# 递归函数调用栈
def fact(x):
    """
    x: int
    """
    if x == 1:
        return 1

    else:
        return x * fact(x-1)

print(fact(3))