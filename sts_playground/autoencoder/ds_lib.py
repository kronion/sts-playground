import tree

class Dataset:
    
    def __init__(self, struct, batch_size):
        self._struct = struct
        self._batch_size = batch_size
        self._size = len(tree.flatten(struct)[0])

    def __iter__(self):
        for start in range(0, self._size, self._batch_size):
            end = start + self._batch_size
            if end >= self._size:
                break
            yield tree.map_structure(lambda xs: xs[start:end], self._struct)
    
    def __len__(self):
        return self._size // self._batch_size
