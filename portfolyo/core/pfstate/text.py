from typing import TYPE_CHECKING
from ..pfline.text import _nestedtree


from ..shared import text

if TYPE_CHECKING:
    from .pfstate import PfState


def pfs_as_string(pfs: PfState, num_of_ts: int, color: bool) -> str:
    lines = ["PfState object."]
    lines.extend(text.index_info(pfs.index))
    spaces = " " * (text.MAX_DEPTH + 5)
    lines.extend([spaces + txtline for txtline in text.dataheader("wqpr")])
    lines.extend(_nestedtree("offtake", pfs.offtakevolume, "wq", num_of_ts))
    lines.extend(_nestedtree("pnl_cost", pfs.pnl_cost, "wqpr", num_of_ts))
    txt = "\n".join(lines)
    return txt if color else text.remove_color(txt)


class PfStateText:
    def __repr__(self):
        return pfs_as_string(self, 5, False)

    def print(self: PfState, num_of_ts: int = 5, color: bool = True) -> None:
        """Treeview of the portfolio state.

        Parameters
        ----------
        num_of_ts : int, optional (default: 5)
            How many timestamps to show for each PfLine.
        color : bool, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output
            devices.

        Returns
        -------
        None
        """
        print(pfs_as_string(self, num_of_ts, color))
