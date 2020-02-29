'''
@Author: your name
@Date: 2020-02-01 11:15:33
@LastEditTime : 2020-02-01 22:23:16
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\单向链表.py
'''
'''
什么是 链表
链表顾名思义就是～链
链表是一种动态数据结构，他的特点是用一组任意的存储单元存放数据元素。
链表中每一个元素成为“结点”，每一个结点都是由数据域和指针域组成的。跟数组不同链表不同预先定义大小，而且硬件支持的话可以无限扩展。

链表与数组的不同点：
数组需要预先定义大小，无法适应数据动态地增减，数据小于定义的长度会浪费内存，数据超过预定义的长度无法插入。
而链表是动态增删数据，可以随意增加。
数组适用于获取元素的操作，直接get索引即可，链表对于获取元素比较麻烦需要从头一直寻找，但是适用与增删，
直接修改节点的指向即可，但是对于数组就比较麻烦了，
例如［1，2，3，4］需要在下标为1的位置插入－2，则需要将［2，3，4］后移，赋值ls[1]=-2
数组从栈中分配空间, 对于程序员方便快速,但自由度小。链表从堆中分配空间, 自由度大但申请管理比较麻烦.

单向链表也叫单链表，是链表中最简单的一种形式，它的每个节点包含两个域，一个信息域（元素域）和一个链接域。
这个链接指向链表中的下一个节点，而最后一个节点的链接域则指向一个空值。

表元素域elem用来存放具体的数据。
链接域next用来存放下一个节点的位置（python中的标识）
变量p指向链表的头节点（首节点）的位置，从p出发能找到表中的任意节点。
'''

class Node(object):
    # 单链表的节点
    def __init__(self, item):
        self.elem = item
        self.next = None


class SingleLinkList(object):
    """单链表"""
    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head == None

    def lenght(self):
        '''链表的长度'''
        # cur游标，用来遍历节点
        cur = self.__head
        count = 0
        while cur != None:
            count += 1
            # cur是Node节点，cur=cur.next将cur.next内存地址传给cur
            cur = cur.next
        return count

    def travel(self):
        '''遍历整个链表'''
        cur = self.__head
        while cur != None:
            print(cur.elem)
            cur = cur.next

        print("")

    def add(self, item):
        '''头部添加'''
        node = Node(item)
        node.next = self.__head
        self.__head = node

    def append(self, item):
        '''尾部添加'''
        node = Node(item)
        if self.__head == None:
            self.__head = node
        else:
            cur = self.__head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        '''指定位置添加元素
        Param: pos 从0开始
        '''
        if pos < 0:
            self.add(item)
        elif pos > (self.lenght() - 1):
            self.append(item)
        else:
            node = Node(item)
            count = 0
            cur = self.__head
            while count < (pos-1):
                count += 1
                cur = cur.next
            node.next = cur.next
            cur.next = node

    def remove(self, item):
        '''删除某个元素'''
        cur = self.__head
        pre = None
        while cur != None:
            if cur.elem == item:
                if pre == None:
                    self.__head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

    def search(self, item):
        '''查找某个元素是否存在'''
        cur = self.__head
        while cur != None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False


class SingleCycleLinkList(object):
    """单链表"""
    def __init__(self, node=None):
        self.__head = node
        if node:
            node.next = node

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head == None

    def lenght(self):
        '''链表的长度'''
        if self.is_empty():
            return 0
        # cur游标，用来遍历节点
        cur = self.__head
        count = 1
        while cur.next != self.__head:
            count += 1
            # cur是Node节点，cur=cur.next将cur.next内存地址传给cur
            cur = cur.next
        return count

    def travel(self):
        '''遍历整个链表'''
        if self.is_empty():
            return
        cur = self.__head
        while cur.next != self.__head:
            print(cur.elem, end=" ")
            cur = cur.next
        # 退出while循环时，指上尾节点，但尾结点未打印
        print(cur.elem)


    def add(self, item):
        '''头部添加'''
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            # 退出while循环时，指上尾节点，但尾结点未打印
            self.__head = node
            node.next = self.__head

    def append(self, item):
        '''尾部添加'''
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            # 退出while循环时，指上尾节点，但尾结点未打印
            node.next = self.__head
            cur.next = node

    def insert(self, pos, item):
        '''指定位置添加元素
        Param: pos 从0开始
        '''
        if pos < 0:
            self.add(item)
        elif pos > (self.lenght() - 1):
            self.append(item)
        else:
            node = Node(item)
            count = 0
            cur = self.__head
            while count < (pos-1):
                count += 1
                cur = cur.next
            # 退出while循环时，指上尾节点，但尾结点未打印
            node.next = cur.next
            cur.next = node

    def remove(self, item):
        '''删除某个元素'''
        if self.is_empty():
            return
        cur = self.__head
        pre = None
        while cur.next != self.__head:
            if cur.elem == item:
                # 先判断是否是头结点
                # 头结点
                if cur == self.__head:
                    # 头结点
                    # 找尾结点
                    rear = self.__head
                    while rear.next == self.__head:
                        rear = rear.next
                    self.__head = cur.next
                    rear.next = self.__head
                else：
                    # 中间节点
                    pre.next = cur.next
                return 0
            else:
                pre = cur
                cur = cur.next
        # 退出while循环时，指上尾节点，但尾结点未打印
        if cur.elem == item:
            if cur == self.__head:
                self.__head = None
            else:
                # pre.next = cur.next
                pre.next = self.__head

    def search(self, item):
        '''查找某个元素是否存在'''
        if self.is_empty():
            return False
        cur = self.__head
        while cur.next != self.__head:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        # 退出while循环时，指上尾节点，但尾结点未打印
        if cur.elem == item:
            return True 
        return False


class DoubleLinkList(object):
    """单链表"""
    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head == None

    def lenght(self):
        '''链表的长度'''
        # cur游标，用来遍历节点
        cur = self.__head
        count = 0
        while cur != None:
            count += 1
            # cur是Node节点，cur=cur.next将cur.next内存地址传给cur
            cur = cur.next
        return count

    def travel(self):
        '''遍历整个链表'''
        cur = self.__head
        while cur != None:
            print(cur.elem)
            cur = cur.next

        print("")

    def add(self, item):
        '''头部添加'''
        node = Node(item)
        node.next = self.__head
        self.__head = node
        node.next.prev = node

    def append(self, item):
        '''尾部添加'''
        node = Node(item)
        if self.__head == None:
            self.__head = node
        else:
            cur = self.__head
            while cur.next != None:
                cur = cur.next
            cur.next = node
            node.prev = cur

    def insert(self, pos, item):
        '''指定位置添加元素
        Param: pos 从0开始
        '''
        if pos < 0:
            self.add(item)
        elif pos > (self.lenght() - 1):
            self.append(item)
        else:
            cur = self.__head
            while count < pos:
                count += 1
                cur = cur.next
            node = Node(item)
            node.next = cur
            node.prev = cur.prev
            cur.prev.next = node
            cur.prev = node

    def remove(self, item):
        '''删除某个元素'''
        cur = self.__head
        while cur != None:
            if cur.elem == item:
                if pre == None:
                    self.__head = cur.next
                    if cur.next:
                        # 判断链表是否只有一个节点
                        cur.next.prev = None
                else:
                    cur.prev.next = cur.next
                    if cur.next:
                        cur.next.prev = cur.prev
                break
            else:
                pre = cur
                cur = cur.next

    def search(self, item):
        '''查找某个元素是否存在'''
        cur = self.__head
        while cur != None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False

if __name__ == "__main__":
    sll = SingleLinkList()
    print("-------")
    print(sll)
    print("-------")
    print(sll.is_empty())
    print(sll.lenght())
    sll.append(100)
    sll.append(200)
    print(sll.travel())
    sll.add(10)
    print(sll.travel())
    sll.insert(3, 99)
    print(sll.travel())
    print(sll.search(99))
    sll.remove(99)
    print(sll.travel())


    dll = DoubleLinkList()
    print("-------")
    print(dll)
    print("-------")
    print(dll.is_empty())
    print(dll.lenght())
    dll.append(100)
    dll.append(200)
    print(dll.travel())
    dll.add(10)
    print(dll.travel())
    dll.insert(3, 99)
    print(dll.travel())
    print(dll.search(99))
    dll.remove(99)
    print(dll.travel())

