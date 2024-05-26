from __future__ import annotations

from typing import TYPE_CHECKING

from ..pfline import text as pfline_text
from ..shared import text as shared_text

if TYPE_CHECKING:
    from .pfstate import PfState


def pfs_as_string(pfs: PfState, num_of_ts: int, color: bool) -> str:
    lines = ["PfState object."]
    lines.extend(shared_text.index_info(pfs.index))
    spaces = " " * (shared_text.MAX_DEPTH + 5)
    lines.extend([spaces + txtline for txtline in shared_text.dataheader("wqpr")])
    lines.extend(pfline_text.nestedtree("offtake", pfs.offtakevolume, "wq", num_of_ts))
    lines.extend(pfline_text.nestedtree("pnl_cost", pfs.pnl_cost, "wqpr", num_of_ts))
    txt = "\n".join(lines)
    return txt if color else shared_text.remove_color(txt)


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
