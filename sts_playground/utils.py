import enum
import typing as tp

class StructType(enum.Enum):
    MAPPING = enum.auto()
    SEQUENCE = enum.auto()
    LEAF = enum.auto()

def to_struct_type(x):
    if isinstance(x, tp.Mapping):
        return StructType.MAPPING
    elif isinstance(x, tp.Sequence):
        return StructType.SEQUENCE
    return StructType.LEAF

Path = tuple[tp.Any, ...]

def tree_diff(t1, t2) -> tp.Iterator[tuple[Path, str]]:
    """Lists differences between two structures """
    if isinstance(t1, tp.Mapping) and isinstance(t2, tp.Mapping):
        for k1 in set(t1).difference(t2):
            yield (k1,), 'in first but not second'
        for k2 in set(t2).difference(t1):
            yield (k2,), 'in second but not first'
        
        for k in set(t1).intersection(t2):
            for path, msg in tree_diff(t1[k], t2[k]):
                yield (k,) + path, msg
    
    elif isinstance(t1, tp.Sequence) and isinstance(t2, tp.Sequence):
        if len(t1) != len(t2):
            yield (), f'lengths {len(t1)} and {len(t2)}'
        elif type(t1) != type(t2):
            yield (), f'types {type(t1)} and {type(t2)}'
        else:
            for i, (v1, v2) in enumerate(zip(t1, t2)):
                for path, msg in tree_diff(v1, v2):
                    yield (i,) + path, msg

    elif to_struct_type(t1) != to_struct_type(t2):
        yield (), f'types {type(t1)} and {type(t2)}'