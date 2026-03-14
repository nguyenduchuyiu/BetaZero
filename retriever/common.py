import re
import json
import torch
import networkx as nx
from loguru import logger
from lean_dojo import Pos
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"


def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


@dataclass(unsafe_hash=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: str
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
            isinstance(self.start, Pos)
            and isinstance(self.end, Pos)
            and self.start <= self.end
        )
        assert isinstance(self.code, str) and self.code != ""

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        annot_full_name = f"{MARK_START_SYMBOL}{self.full_name}{MARK_END_SYMBOL}"
        code = self.code.replace(f"_root_.{self.full_name}", annot_full_name)
        fields = self.full_name.split(".")

        for i in range(len(fields)):
            prefix = ".".join(fields[i:])
            new_code = re.sub(f"(?<=\\s)«?{prefix}»?", annot_full_name, code)
            if new_code != code:
                code = new_code
                break

        return code


class PremiseSet:
    """A set of premises indexed by their paths and full names."""

    path2premises: Dict[str, Dict[str, Premise]]

    def __init__(self) -> None:
        self.path2premises = {}

    def __iter__(self) -> Generator[Premise, None, None]:
        for _, premises in self.path2premises.items():
            for p in premises.values():
                yield p

    def add(self, p: Premise) -> None:
        if p.path in self.path2premises:
            self.path2premises[p.path][p.full_name] = p
        else:
            self.path2premises[p.path] = {p.full_name: p}

    def update(self, premises: List[Premise]) -> None:
        for p in premises:
            self.add(p)

    def __contains__(self, p: Premise) -> bool:
        return (
            p.path in self.path2premises and p.full_name in self.path2premises[p.path]
        )

    def __len__(self) -> int:
        return sum(len(premises) for premises in self.path2premises.values())


@dataclass(frozen=True)
class File:
    """A file defines 0 or multiple premises."""

    path: str
    """Path of the ``*.lean`` file.
    """

    premises: List[Premise] = field(repr=False, compare=False)
    """A list of premises defined in this file.
    """

    @classmethod
    def from_data(cls, file_data: Dict[str, Any]) -> "File":
        """Construct a :class:`File` object from ``file_data``."""
        path = file_data["path"]
        premises = []
        for p in file_data["premises"]:
            full_name = p["full_name"]
            if full_name is None:
                continue
            if "user__.n" in full_name or p["code"] == "":
                # Ignore ill-formed premises (often due to errors in ASTs).
                continue
            if full_name.startswith("[") and full_name.endswith("]"):
                # Ignore mutual definitions.
                continue
            premises.append(
                Premise(
                    path, p["full_name"], Pos(*p["start"]), Pos(*p["end"]), p["code"]
                )
            )
        return cls(path, premises)

    @property
    def is_empty(self) -> bool:
        """Check whether the file contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files. 
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    def __init__(self, jsonl_path: str) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        dep_graph = nx.DiGraph()
        self.all_premises = []

        logger.info(f"Building the corpus from {jsonl_path}")

        for line in open(jsonl_path):
            file_data = json.loads(line)
            path = file_data["path"]
            assert not dep_graph.has_node(path)
            file = File.from_data(file_data)

            dep_graph.add_node(path, file=file)
            self.all_premises.extend(file.premises)

            for p in file_data["imports"]:
                assert dep_graph.has_node(p)
                dep_graph.add_edge(path, p)

        assert nx.is_directed_acyclic_graph(dep_graph)
        self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

        self.imported_premises_cache = {}
        self.fill_cache()

    def _get_file(self, path: str) -> File:
        return self.transitive_dep_graph.nodes[path]["file"]

    def __len__(self) -> int:
        return len(self.all_premises)

    def __contains__(self, path: str) -> bool:
        return path in self.transitive_dep_graph

    def __getitem__(self, idx: int) -> Premise:
        return self.all_premises[idx]

    @property
    def files(self) -> List[File]:
        return [self._get_file(p) for p in self.transitive_dep_graph.nodes]

    @property
    def num_files(self) -> int:
        return len(self.files)

    def get_dependencies(self, path: str) -> List[str]:
        """Return a list of (direct and indirect) dependencies of the file ``path``."""
        return list(self.transitive_dep_graph.successors(path))

    def get_premises(self, path: str) -> List[Premise]:
        """Return a list of premises defined in the file ``path``."""
        return self._get_file(path).premises

    def num_premises(self, path: str) -> int:
        """Return the number of premises defined in the file ``path``."""
        return len(self.get_premises(path))

    def locate_premise(self, path: str, pos: Pos) -> Optional[Premise]:
        """Return a premise at position ``pos`` in file ``path``.

        Return None if no such premise can be found.
        """
        for p in self.get_premises(path):
            assert p.path == path
            if p.start <= pos <= p.end:
                return p
        return None

    def fill_cache(self) -> None:
        for path in self.transitive_dep_graph.nodes:
            self._get_imported_premises(path)

    def _get_imported_premises(self, path: str) -> List[Premise]:
        """Return a list of premises imported in file ``path``. The result is cached."""
        premises = self.imported_premises_cache.get(path, None)
        if premises is not None:
            return premises

        premises = []
        for p in self.transitive_dep_graph.successors(path):
            premises.extend(self._get_file(p).premises)
        self.imported_premises_cache[path] = premises
        return premises

    def get_accessible_premises(self, path: str, pos: Pos) -> PremiseSet:
        """Return the set of premises accessible at position ``pos`` in file ``path``,
        i.e., all premises defined in the (transitively) imported files or earlier in the same file.
        """
        premises = PremiseSet()
        for p in self.get_premises(path):
            if p.end <= pos:
                premises.add(p)
        premises.update(self._get_imported_premises(path))
        return premises

    def get_accessible_premise_indexes(self, path: str, pos: Pos) -> List[int]:
        return [
            i
            for i, p in enumerate(self.all_premises)
            if (p.path == path and p.end <= pos)
            or self.transitive_dep_graph.has_edge(path, p.path)
        ]


@dataclass(frozen=True)
class IndexedCorpus:
    """A corpus with premise embeddings."""

    corpus: Corpus
    embeddings: torch.FloatTensor

    def __post_init__(self):
        assert self.embeddings.device == torch.device("cpu")
        assert len(self.embeddings) == len(self.corpus)


def get_all_pos_premises(annot_tac, corpus: Corpus) -> List[Premise]:
    """Return a list of all premises that are used in the tactic ``annot_tac``."""
    _, provenances = annot_tac
    all_pos_premises = set()

    for prov in provenances:
        def_path = prov["def_path"]
        p = corpus.locate_premise(def_path, Pos(*prov["def_pos"]))
        if p is not None:
            all_pos_premises.add(p)
        else:
            logger.warning(f"Cannot locate premise: {prov}")

    return list(all_pos_premises)


