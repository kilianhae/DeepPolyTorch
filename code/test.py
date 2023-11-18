class Grandparent:
    def __init__(self):
        print("Grandparent init")

class Parent(Grandparent):
    def __init__(self):
        print("Parent init")
        super().__init__()

class Child(Parent):
    def __init__(self):
        print("Child init")
        super().__init__()

c = Child() 