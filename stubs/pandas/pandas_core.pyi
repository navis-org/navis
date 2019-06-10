from typing import *
from typing_extensions import Literal

from numpy import ndarray

"""
Gotchas:
- Sequence[str] = str
- List[Union[int, str]] != List[int]
- Sequence[Union[int, str]] = Sequence[int]
- "= ..." is necessary to define as optional argument
- overloaded functions don't need the implementation in stubs
- Literal[True/False] must not have default values
- when using Literal[True] and Literal[False], we also need an overloaded
  function with just "bool" for cases were we don't
"""


class DataFrame:
    columns: 'ndarray'

    T: 'DataFrame'

    loc: 'FrameLoc'
    iloc: 'FrameiLoc'
    at: 'At'

    values: ndarray
    shape: Sequence[int]

    empty: bool

    def __init__(self,
                 data: Optional[Union[ndarray,
                                      Iterable,
                                      dict,
                                      'DataFrame']] = ...,
                 index: Iterable = ...,
                 columns: Iterable = ...,
                 dtype: Any = ...,
                 copy: bool = ...,
                 ): ...

    @overload
    def __getitem__(self,
                    key: Union[str, int]) -> 'Series': ...

    @overload
    def __getitem__(self,
                    key: Union[List[str],  # must not be Sequence as Sequence[str] == str
                               Sequence[int],
                               Sequence[bool],
                               Iterable[int],
                               Iterable[bool],
                               Series,
                               ]) -> 'DataFrame': ...

    def __setitem__(self,
                    key: Union[str, int, bool,
                               List[Union[str, int, bool]],
                               ],
                    value) -> None: ...

    def __contains__(self, key: str) -> bool: ...

    def __getattr__(self,
                    key: str) -> 'Series': ...

    def __len__(self) -> int: ...

    def __invert__(self) -> 'DataFrame': ...

    def __sub__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __add__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __mul__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __pow__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __div__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __truediv__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __eq__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ne__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __lt__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __le__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __gt__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ge__(self, other: Any) -> 'DataFrame': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    @property
    def index(self) -> 'Index': ...

    @index.setter
    def index(self, values=Union['Index', Sequence[str]]): ...

    @overload
    def set_index(self,
                  keys: str,
                  inplace: Literal[True],
                  drop: bool = ...
                  ) -> None: ...

    @overload
    def set_index(self,
                  keys: str,
                  inplace: Literal[False],
                  drop: bool = ...
                  ) -> 'DataFrame': ...

    @overload
    def reset_index(self,
                    inplace: Literal[True],
                    drop: bool = ...
                    ) -> None: ...

    @overload
    def reset_index(self,
                    inplace: Literal[False],
                    drop: bool = ...
                    ) -> 'DataFrame': ...

    @overload
    def melt(self,
             id_vars: Union[str, List[str]],
             inplace: Literal[True],
             drop: bool = ...
             ) -> None: ...

    @overload
    def melt(self,
             id_vars: Union[str, List[str]],
             inplace: Literal[False],
             drop: bool = ...
             ) -> 'DataFrame': ...

    @overload
    def drop(self,
             labels: Union[str, Sequence[str]],
             inplace: Literal[True],
             axis: int = 0) -> None: ...

    @overload
    def drop(self,
             labels: Union[str, Sequence[str]],
             inplace: Literal[False],
             axis: int = 0) -> 'DataFrame': ...

    def sum(self,
            axis: int = ...,
            **kwargs) -> 'DataFrame': ...

    def min(self,
            axis: int = ...,
            **kwargs) -> 'DataFrame': ...

    def max(self,
            axis: int = ...,
            **kwargs) -> 'DataFrame': ...

    def mean(self,
             axis: int = ...,
             **kwargs) -> 'DataFrame': ...

    def duplicated(self,
                   subset: Union[Sequence[Union[str, int]], int, str] = ...,
                   keep: Union[Sequence[Union[str, int]], int, str] = ...
                   ) -> Series: ...

    @overload
    def sort_values(self,
                    by: Union[str, List[str]],
                    inplace: Literal[True],
                    ascending: bool = ...) -> None: ...

    @overload
    def sort_values(self,
                    by: Union[str, List[str]],
                    inplace: Literal[False],
                    ascending: bool = ...) -> 'DataFrame': ...

    @overload
    def where(self,
              cond: Union[Sequence[bool], Callable],
              other: Union[str, int, None],
              inplace: Literal[True]) -> None: ...

    @overload
    def where(self,
              cond: Union[Sequence[bool], Callable],
              other: Union[str, int, None],
              inplace: Literal[False]) -> 'DataFrame': ...

    def describe(self) -> 'DataFrame': ...

    def itertuples(self) -> Iterator['Series']: ...

    def copy(self) -> 'DataFrame': ...

    def groupby(self, by: Union[str, List[str]], **kwargs) -> 'DataFrame': ...

    def to_json(self, **kwargs) -> str: ...

    def isnull(self) -> 'DataFrame': ...

    def any(self, axis: int) -> 'Series': ...

    def to_dict(self, **kwargs) -> Dict[str, dict]: ...

    def astype(self, Any) -> 'DataFrame': ...

    def to_matrix(self, **kwargs) -> 'ndarray': ...

    def as_matrix(self, **kwargs) -> Sequence: ...

    def apply(self, x: Callable, **kwargs) -> 'DataFrame': ...


class Series:

    # These are custom values
    node_id: 'Series'
    parent_id: 'Series'
    relation: 'Series'

    T: 'DataFrame'

    loc: 'SeriesLoc'
    at: 'At'

    name: str

    values: ndarray
    shape: Sequence[int]

    empty: bool

    def __init__(self,
                 data: Union[ndarray,
                             Iterable,
                             dict,
                             'Series'],
                 index: Iterable = ...,
                 dtype: Any = ...,
                 copy: bool = ...,
                 ): ...

    def __getitem__(self, idx) -> ndarray: ...

    def __setitem__(self, idx, value) -> None: ...

    def __getattr__(self, key: str) -> Union[int, str, float]: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator: ...

    def __invert__(self) -> 'Series': ...

    def __eq__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ne__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __lt__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __le__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __gt__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ge__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __sub__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __add__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __mul__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __pow__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __div__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __truediv__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __or__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __xor__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __and__(self, other: Any) -> 'Series': ...  # type: ignore  # already defined in typeshed - does not like being overriden

    @property
    def index(self) -> 'Index': ...

    @index.setter
    def index(self, values=Union['Index', Sequence[str]]): ...

    def sum(self) -> float: ...

    def min(self) -> float: ...

    def max(self) -> float: ...

    def mean(self) -> float: ...

    def map(self, arg: Union[Callable, dict]) -> Series: ...

    def isin(self, arg: Union[Iterable, 'Series']) -> 'Series': ...

    def copy(self) -> 'Series': ...

    def astype(self, Any, **kwargs) -> 'Series': ...

    def isnull(self) -> Sequence[bool]: ...

    def duplicated(self, **kwargs) -> 'Series': ...

    def to_dict(self, **kwargs) -> Dict[str, Any]: ...

    def tolist(self, **kwargs) -> list: ...

    def apply(self, x: Callable, **kwargs) -> 'Series': ...

    def unique(self, **kwargs) -> 'ndarray': ...


class FrameLoc:
    def __init__(self): ...

    @overload
    def __getitem__(self,
                    key: Tuple[Union[str, int],
                               Union[str, int]]) -> Union[float, str]: ...

    @overload
    def __getitem__(self,
                    key: Union[str, int,
                               Tuple[Union[List[str], List[int], List[bool], 'Series', 'DataFrame'],
                                     Union[str, int, bool]],
                               Tuple[Union[str, int, bool],
                                     Union[List[str], List[int], List[bool], 'Series', 'DataFrame']]
                               ]) -> 'Series': ...

    @overload
    def __getitem__(self,
                    key: Union[Union[List[str], List[int], List[bool], 'Series', 'DataFrame'],
                               Tuple[Union[List[str], List[int], List[bool], 'Series', 'DataFrame', slice],
                                     Union[List[str], List[int], List[bool], 'Series', 'DataFrame', slice]]
                               ]) -> 'DataFrame': ...

    def __setitem__(self,
                    key1: Any,
                    key2: Optional[Any]) -> None: ...


class FrameiLoc:
    def __init__(self): ...

    @overload
    def __getitem__(self,
                    key: Tuple[int,
                               int]) -> Union[float, str]: ...

    @overload
    def __getitem__(self,
                    key: Union[int,
                               Tuple[Union[List[int], List[bool], 'Series', slice],
                                     Union[int, bool]],
                               Tuple[Union[int, bool, slice],
                                     Union[List[int], List[bool], 'Series']]
                               ]) -> 'Series': ...

    @overload
    def __getitem__(self,
                    key: Union[Union[List[int], List[bool], 'Series'],
                               Tuple[Union[List[int], List[bool], 'Series'],
                                     Union[List[int], List[bool], 'Series']]
                               ]) -> 'DataFrame': ...


class SeriesLoc:
    def __init__(self): ...

    @overload
    def __getitem__(self,
                    key: Union[str, int]) -> Union[str, float]: ...

    @overload
    def __getitem__(self,
                    key: List[Union[str, int, bool, 'Series']]) -> ndarray: ...

    def __eq__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ne__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __lt__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __le__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __gt__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ge__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden


class At():
    def __init__(self): ...

    def __getitem__(self,
                    key: Tuple[Union[str, int],
                               Union[str, int]]) -> Union[str, float]: ...

    def __setitem__(self,
                    key: Tuple[Union[str, int],
                               Union[str, int]],
                    value: Any) -> None: ...


class Index(Series):
    def __init__(self): ...

    @overload
    def set_names(self,
                  names: Union[str, List[Union[str, int]]],
                  inplace: Literal[True]) -> None: ...

    @overload
    def set_names(self,
                  names: Union[str, List[Union[str, int]]],
                  inplace: Literal[False]) -> Index: ...


def read_json(x: str, **kwargs) -> 'DataFrame': ...


def concat(x: Sequence[DataFrame], **kwargs) -> 'DataFrame': ...
