'''
@Author: your name
@Date: 2020-02-02 14:13:33
@LastEditTime : 2020-02-02 14:48:01
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\栈和队列.py
'''

class Stack(object):
    """队列"""
    def __init__(self):
        self.__list = []

    def is_empty(self):
        return self.__list == []

    def push(self, item):
        self.__list.append(item)
    
    def pop(self):
        return self.__list.pop()
    
    def peek(self):
        return self.__list[len(self.__list)-1]
    def size(self):
        return len(self.__list)


class Queue(object):
    """队列"""
    def __init__(self):
        self.__list = []

    def is_empty(self):
        return self.__list == []

    def enqueue(self,item):
        """往队列中添加一种元素"""
        self.__list.append(item)

    def dequeue(self):
        """"弹出元素"""
        return self.__list.pop(0)
        
    def size(self):
        return len(self.__list)


if __name__ == "__main__":
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    s.push(5)
    s.push(6)
    s.push(7)

    print(s.peek())
    print(s.size())
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())

    q = Queue()
    
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)
    q.enqueue(5)
    q.enqueue(6)
    q.enqueue(7)

    print(q.is_empty())
    print(q.size())

    print(q)