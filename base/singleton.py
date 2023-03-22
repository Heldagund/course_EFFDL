def singleton(cls):
    _instance = {}

    def inner(*arg, **karg):
        if cls not in _instance:
            _instance[cls] = cls(*arg, **karg)
        return _instance[cls]
    return inner

# class singleton(object):
#     def __init__(self, cls):
#         self._cls = cls
#         self._instance = {}
#     def __call__(self):
#         if self._cls not in self._instance:
#             self._instance[self._cls] = self._cls()
#         return self._instance[self._cls]