class OneTwoFunction(object):
    '''Allows a function to be called that returns a tuple. '''

    def __init__(self, function):
        self.function = function
        self.nextCall = 1
        self.result = None

    def one(self, value):
        if self.nextCall != 1:
            raise Exception('Wrong call order; was expecting %d' % self.nextCall)
        self.lastValue = value
        self.result = self.function(value)
        self.nextCall = 2
        return self.result[0]

    def two(self, value):
        if self.nextCall != 2:
            raise Exception('Wrong call order; was expecting %d' % self.nextCall)
        if value != self.lastValue:
            raise Exception('Called two with a different value.')
        self.nextCall = 1
        return self.result[1]
