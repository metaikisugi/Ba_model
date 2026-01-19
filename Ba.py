import argparse
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bisect


# ============================================================
# Utility: stable colors per "true vertex"
# ============================================================
def color_for_int(i: int):
    rnd = (i * 2654435761) & 0xFFFFFFFF
    r = ((rnd >> 0) & 255) / 255
    g = ((rnd >> 8) & 255) / 255
    b = ((rnd >> 16) & 255) / 255
    return (0.15 + 0.75 * r, 0.15 + 0.75 * g, 0.15 + 0.75 * b)


def true_vertex_of(t: int, m: int) -> int:
    """v(t)=ceil(t/m)"""
    return (t - 1) // m + 1


# ============================================================
# Simple incremental spring layout (for true-vertex graph)
# ============================================================
def spring_layout_incremental(
    n: int,
    edges: List[Tuple[int, int]],
    pos: np.ndarray,
    iters: int = 8,
    seed: int = 0,
    step: float = 0.02,
):
    rng = np.random.default_rng(seed)
    if pos is None or pos.shape != (n, 2):
        pos = rng.normal(scale=0.5, size=(n, 2))

    k = 1.0 / np.sqrt(max(1, n))

    for _ in range(iters):
        disp = np.zeros_like(pos)

        # repulsion O(n^2)
        for i in range(n):
            delta = pos[i] - pos
            dist2 = (delta[:, 0] ** 2 + delta[:, 1] ** 2) + 1e-9
            dist = np.sqrt(dist2)
            rep = (k * k) / dist
            disp[i] += np.sum((delta / dist[:, None]) * rep[:, None], axis=0)

        # attraction on edges
        for a, b in edges:
            if a == b:
                continue
            delta = pos[a] - pos[b]
            dist = np.sqrt(delta[0] ** 2 + delta[1] ** 2) + 1e-9
            att = (dist * dist) / k
            vec = (delta / dist) * att
            disp[a] -= vec
            disp[b] += vec

        pos = pos + step * disp
        pos = np.clip(pos, -1.2, 1.2)

    return pos


# ============================================================
# Degree computation on true-vertex graph
# ============================================================
def degrees_from_edges(n_true: int, edges_true: List[Tuple[int, int]]) -> List[int]:
    deg = [0] * (n_true + 1)
    for a, b in edges_true:
        if 1 <= a <= n_true and 1 <= b <= n_true:
            deg[a] += 1
            deg[b] += 1
    return deg


# ============================================================
# HALF model (your explicit L/R endpoints with "jump if pick R")
# ============================================================
@dataclass
class Endpoint:
    kind: str            # "L" or "R"
    t: int               # chord index
    idnum: int           # L_t=2t-1, R_t=2t
    root_vertex: int     # color by this true vertex


@dataclass
class HalfFrame:
    t: int
    m: int
    n_true: int
    picked: Endpoint
    receiver_left_t: int
    edges_true: List[Tuple[int, int]]
    pool_view: List[Endpoint]


def simulate_half_model(m: int, n_true: int, seed: int, pool_show: int = 260):
    """
    At step t:
      - new endpoints L_t (circle), R_t (square)
      - R_t picks uniformly from all previous endpoints in pool
      - If picks L_i: receiver is L_i
      - If picks R_j: receiver jumps to L_{p(j)} (the left that accepted R_j)
      - Add edge on true vertices: v(t) -> v(receiver)
    """
    rng = random.Random(seed)
    N = m * n_true

    # partner_left[t] = i means R_t is accepted by L_i
    partner_left = [0] * (N + 1)

    pool: List[Endpoint] = []
    edges_true: List[Tuple[int, int]] = []
    frames: List[HalfFrame] = []

    # init t=1 to avoid empty pool
    partner_left[1] = 1
    v1 = true_vertex_of(1, m)
    L1 = Endpoint("L", 1, 1, root_vertex=v1)
    R1 = Endpoint("R", 1, 2, root_vertex=v1)
    pool.extend([L1, R1])

    for t in range(2, N + 1):
        picked = pool[rng.randrange(0, len(pool))]

        if picked.kind == "L":
            receiver_left_t = picked.t
        else:
            receiver_left_t = partner_left[picked.t]  # jump to its receiver

        partner_left[t] = receiver_left_t

        from_v = true_vertex_of(t, m)
        to_v = true_vertex_of(receiver_left_t, m)
        edges_true.append((from_v, to_v))

        Lt = Endpoint("L", t, 2 * t - 1, root_vertex=true_vertex_of(t, m))
        Rt = Endpoint("R", t, 2 * t, root_vertex=true_vertex_of(receiver_left_t, m))
        pool.extend([Lt, Rt])

        pool_view = pool[-pool_show:] if len(pool) > pool_show else pool[:]

        frames.append(
            HalfFrame(
                t=t,
                m=m,
                n_true=n_true,
                picked=picked,
                receiver_left_t=receiver_left_t,
                edges_true=edges_true.copy(),
                pool_view=pool_view.copy(),
            )
        )

    pair_list = [(t, partner_left[t]) for t in range(1, N + 1)]  # (t, receiver_left_t)
    return frames, pair_list


def animate_half(frames: List[HalfFrame], pair_list, m: int, n_true: int, seed: int, fps: int, print_all: bool):
    interval = int(1000 / fps)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.5], height_ratios=[1.0, 1.0])

    ax_pool = fig.add_subplot(gs[0, 0])
    ax_deg = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[:, 1])

    pos = np.random.default_rng(seed).normal(scale=0.7, size=(n_true + 1, 2))
    printed_until = 1

    def print_pairs_up_to(t: int):
        nonlocal printed_until
        if print_all:
            print(f"\n=== HALF pair list up to t={t} ===")
            for s in range(1, t + 1):
                rec = pair_list[s - 1][1]
                print(
                    f"t={s:3d}:  R_{s}(id={2*s:3d}) -> L_{rec}(id={2*rec-1:3d})"
                    f" | true v({s})={true_vertex_of(s,m)} -> v({rec})={true_vertex_of(rec,m)}"
                )
            print("=== end ===\n", flush=True)
            printed_until = t + 1
        else:
            while printed_until <= t:
                s = printed_until
                rec = pair_list[s - 1][1]
                print(
                    f"t={s:3d}:  R_{s}(id={2*s:3d}) -> L_{rec}(id={2*rec-1:3d})"
                    f" | true v({s})={true_vertex_of(s,m)} -> v({rec})={true_vertex_of(rec,m)}",
                    flush=True
                )
                printed_until += 1

    def draw_pool(ax, pool_view: List[Endpoint], picked: Endpoint, receiver_left_t: int, t: int):
        ax.clear()
        ax.set_title("HALF pool (L=circle receptor, R=square stabber; color=true vertex)", fontsize=11)

        xs = np.arange(len(pool_view))
        L_idx = [i for i, e in enumerate(pool_view) if e.kind == "L"]
        R_idx = [i for i, e in enumerate(pool_view) if e.kind == "R"]
        cols = [color_for_int(e.root_vertex) for e in pool_view]

        if L_idx:
            ax.scatter(xs[L_idx], np.zeros(len(L_idx)),
                       s=42, c=[cols[i] for i in L_idx],
                       marker="o", alpha=0.85, edgecolors="none", label="L")
        if R_idx:
            ax.scatter(xs[R_idx], np.zeros(len(R_idx)),
                       s=55, c=[cols[i] for i in R_idx],
                       marker="s", alpha=0.85, edgecolors="none", label="R")

        # highlight picked if visible
        picked_pos = None
        for i, e in enumerate(pool_view):
            if (e.kind == picked.kind) and (e.t == picked.t) and (e.idnum == picked.idnum):
                picked_pos = i
                break
        if picked_pos is not None:
            ax.scatter([picked_pos], [0], s=220, facecolors="none", edgecolors="black",
                       linewidths=2.5, zorder=10)

        ax.set_xlim(-1, max(10, len(pool_view)))
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel(f"shown tail of pool (pool_show={len(pool_view)})")
        ax.legend(loc="upper right", fontsize=9)

        ax.text(
            0.02, 0.90,
            f"step t={t}\n"
            f"picked={picked.kind}_{picked.t} (id={picked.idnum})\n"
            f"receiver=L_{receiver_left_t}\n"
            f"new stabber=R_{t} (id={2*t})\n"
            f"true edge: v(t) -> v(receiver)",
            transform=ax.transAxes,
            fontsize=9, va="top"
        )

    def draw_degrees(ax, edges_true: List[Tuple[int, int]], t: int):
        ax.clear()
        ax.set_title("Degree distribution on TRUE vertices (after contraction)", fontsize=11)
        ax.set_xlabel("true vertex id (1..n)")
        ax.set_ylabel("degree")

        deg = degrees_from_edges(n_true, edges_true)
        xs = np.arange(1, n_true + 1)
        cols = [color_for_int(i) for i in xs]
        bars = ax.bar(xs, [deg[i] for i in xs], color=cols, edgecolor="none")

        active_max = min(n_true, true_vertex_of(t, m))
        for i in range(active_max + 1, n_true + 1):
            bars[i - 1].set_color((0.85, 0.85, 0.85))

        ax.set_ylim(0, max(1, max(deg) + 2))

    def draw_true_graph(ax, edges_true: List[Tuple[int, int]], t: int):
        nonlocal pos
        ax.clear()
        ax.set_title("True-vertex graph (spring layout)", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        active_max = min(n_true, true_vertex_of(t, m))

        edges0 = [(a - 1, b - 1) for (a, b) in edges_true if 1 <= a <= n_true and 1 <= b <= n_true]
        pos0 = pos[1:].copy()
        pos0 = spring_layout_incremental(n=n_true, edges=edges0, pos=pos0, iters=7, seed=seed, step=0.02)
        pos[1:] = pos0

        for a, b in edges_true:
            if a > active_max or b > active_max:
                continue
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.plot([xa, xb], [ya, yb], lw=1.0, alpha=0.45)

        xs = [pos[i, 0] for i in range(1, active_max + 1)]
        ys = [pos[i, 1] for i in range(1, active_max + 1)]
        cols = [color_for_int(i) for i in range(1, active_max + 1)]
        ax.scatter(xs, ys, s=140, c=cols, edgecolors="k", linewidths=0.4, zorder=3)

        for i in range(1, active_max + 1):
            ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=9, ha="center", va="center", zorder=4)

        # highlight newest edge
        if edges_true:
            a, b = edges_true[-1]
            if a <= active_max and b <= active_max:
                ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], lw=3.0, alpha=0.9)

    def update(k):
        fr = frames[k]
        t = fr.t

        print_pairs_up_to(t)
        draw_pool(ax_pool, fr.pool_view, fr.picked, fr.receiver_left_t, t)
        draw_degrees(ax_deg, fr.edges_true, t)
        draw_true_graph(ax_graph, fr.edges_true, t)

        fig.suptitle(f"[HALF] t={t}/{m*n_true}  (terminal prints R_t -> L_receiver)", fontsize=14)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani


@dataclass
class StandardBAFrame:
    t: int
    n_true: int
    edges: List[Tuple[int, int]]
    degrees: List[int]
    probabilities: List[float]
    selected_nodes: List[int]
    new_node: int


def simulate_standard_ba_model(m: int, n_true: int, seed: int):
    """
    Standard Barabási-Albert model:
    - Start with m initial nodes, fully connected.
    - Each new node adds m edges to existing nodes with probability proportional to their degree.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize with m vertices fully connected
    nodes = list(range(1, m + 1))
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append((nodes[i], nodes[j]))
    
    frames = []
    
    # Current degrees
    deg = [0] * (n_true + 1)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
        
    for t in range(m + 1, n_true + 1):
        # Calculate probabilities
        total_deg = sum(deg)
        if total_deg == 0:
            probs = [1.0/len(nodes)] * len(nodes)
        else:
            probs = [deg[i] / total_deg for i in nodes]
            
        # Select m distinct targets
        targets = rng.choice(nodes, size=m, replace=False, p=probs).tolist()
        
        for target in targets:
            edges.append((t, target))
            deg[t] += 1
            deg[target] += 1
            
        nodes.append(t)
        
        # Save snapshot before next step
        frames.append(
            StandardBAFrame(
                t=t,
                n_true=n_true,
                edges=edges.copy(),
                degrees=deg.copy(),
                probabilities=probs + [0.0] * (n_true - len(nodes) + 1),
                selected_nodes=targets,
                new_node=t
            )
        )
        
    pair_list = []
    for fr in frames:
        for target in fr.selected_nodes:
            pair_list.append((fr.new_node, target))
            
    return frames, pair_list


def animate_standard(frames: List[StandardBAFrame], pair_list, m: int, n_true: int, seed: int, fps: int, print_all: bool):
    interval = int(1000 / fps)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.5], height_ratios=[1.0, 1.0])

    ax_info = fig.add_subplot(gs[0, 0])
    ax_deg = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[:, 1])

    pos = np.random.default_rng(seed).normal(scale=0.7, size=(n_true + 1, 2))
    printed_until = m + 1
    
    def print_edges_up_to(t: int):
        nonlocal printed_until
        if print_all:
            print(f"\n=== STANDARD BA edge list up to t={t} ===")
            for s in range(m + 1, t + 1):
                # Find all edges added at step s
                targets = [target for (node, target) in pair_list if node == s]
                targets_str = ", ".join(str(tgt) for tgt in targets)
                print(f"t={s:3d}:  new node {s} -> [{targets_str}]")
            print("=== end ===\n", flush=True)
            printed_until = t + 1
        else:
            while printed_until <= t:
                s = printed_until
                targets = [target for (node, target) in pair_list if node == s]
                targets_str = ", ".join(str(tgt) for tgt in targets)
                print(f"t={s:3d}:  new node {s} -> [{targets_str}]", flush=True)
                printed_until += 1
    
    def draw_info(ax, fr: StandardBAFrame):
        ax.clear()
        ax.set_title("STANDARD BA: Current step info (P ∝ degree)", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        # Display current step information
        info_text = (
            f"Step t = {fr.new_node}\n\n"
            f"New node: {fr.new_node}\n"
            f"Attaches to (m={m}): {fr.selected_nodes}\n\n"
            f"Selection method:\n"
            f"  P(v) ∝ degree(v)\n\n"
        )
        
        # Show degrees of selected nodes
        info_text += "Selected nodes' degrees:\n"
        for target in fr.selected_nodes:
            deg = fr.degrees[target] - 1  # -1 because we already added the edge
            info_text += f"  node {target}: deg={deg}\n"
        
        ax.text(
            0.05, 0.95,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="monospace"
        )

    def draw_degrees(ax, fr: StandardBAFrame):
        ax.clear()
        ax.set_title("Degree distribution", fontsize=11)
        ax.set_xlabel("vertex id")
        ax.set_ylabel("degree")
        
        xs = np.arange(1, n_true + 1)
        cols = [color_for_int(i) for i in xs]
        # Highlight selected nodes
        edge_cols = ['black' if i in fr.selected_nodes else 'none' for i in xs]
        lws = [2.0 if i in fr.selected_nodes else 0 for i in xs]
        
        ax.bar(xs, fr.degrees[1:n_true+1], color=cols, edgecolor=edge_cols, linewidth=lws)
        ax.set_ylim(0, max(5, max(fr.degrees) + 2))

    def draw_graph(ax, fr: StandardBAFrame):
        nonlocal pos
        ax.clear()
        ax.set_title(f"Standard BA Graph (edges highlighted)", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        active_max = fr.t
        edges0 = [(a - 1, b - 1) for (a, b) in fr.edges if a <= n_true and b <= n_true]
        pos0 = pos[1:].copy()
        pos0 = spring_layout_incremental(n=n_true, edges=edges0, pos=pos0, iters=5, seed=seed, step=0.02)
        pos[1:] = pos0

        for a, b in fr.edges:
            if a > active_max or b > active_max: continue
            xa, ya = pos[a]
            xb, yb = pos[b]
            is_new = (a == fr.new_node and b in fr.selected_nodes) or (b == fr.new_node and a in fr.selected_nodes)
            ax.plot([xa, xb], [ya, yb], lw=2.5 if is_new else 1.0, alpha=0.9 if is_new else 0.4)

        xs = [pos[i, 0] for i in range(1, active_max + 1)]
        ys = [pos[i, 1] for i in range(1, active_max + 1)]
        cols = [color_for_int(i) for i in range(1, active_max + 1)]
        ax.scatter(xs, ys, s=140, c=cols, edgecolors="k", linewidths=0.4, zorder=3)

        for i in range(1, active_max + 1):
            ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=9, ha="center", va="center", zorder=4)

    def update(k):
        fr = frames[k]
        print_edges_up_to(fr.t)
        draw_info(ax_info, fr)
        draw_degrees(ax_deg, fr)
        draw_graph(ax_graph, fr)
        fig.suptitle(f"[STANDARD BA] t={fr.t}/{n_true}  (terminal prints: node -> targets)", fontsize=14)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani


# ============================================================
# LCD model ([0,1] method, sort right endpoints, block-contraction)
# ============================================================
@dataclass
class LCDFrame:
    t: int
    m: int
    n_true: int
    # right endpoints
    R_sorted: List[float]          # R_1..R_N
    W: List[float]                 # W_i = R_{mi} for i=1..n
    # current chord (in rank t)
    r_t: float
    l_t: float
    receiver_k: int                # k = first index with R_k >= l_t
    edges_true: List[Tuple[int, int]]


def simulate_lcd_model(m: int, n_true: int, seed: int):
    """
    Implements Section 6 style generation:

    - N=mn chords.
    - sample r_i ~ M2(0,1): r = sqrt(U)
    - sample l_i | r_i ~ Uniform(0, r_i)
    - sort r_i to obtain R_1..R_N
    - for rank t (1..N), chord is the one whose r_i = R_t
    - receiver k = first index such that R_k >= l_i  (so k <= t)
    - add edge on true vertices: v(t) -> v(k)
    """
    rng = np.random.default_rng(seed)
    N = m * n_true

    U = rng.random(N)
    r = np.sqrt(U)  # density 2x on (0,1)
    l = rng.random(N) * r  # uniform in [0, r_i]

    # sort r, keep permutation
    ord_idx = np.argsort(r)
    R_sorted = r[ord_idx].tolist()

    # W_i = R_{m*i} (1-index), with W_0=0
    W = []
    for i in range(1, n_true + 1):
        W.append(R_sorted[m * i - 1])

    edges_true: List[Tuple[int, int]] = []
    frames: List[LCDFrame] = []

    # for each rank t, get chord index idx = ord_idx[t-1]
    for t in range(1, N + 1):
        idx = ord_idx[t - 1]
        r_t = float(r[idx])
        l_t = float(l[idx])

        # receiver k = lower_bound of l_t in R_sorted
        k = bisect.bisect_left(R_sorted, l_t) + 1  # to 1-index
        if k < 1:
            k = 1
        if k > t:
            # theoretically k <= t because R_t=r_t>=l_t, but safe guard
            k = t

        from_v = true_vertex_of(t, m)
        to_v = true_vertex_of(k, m)
        edges_true.append((from_v, to_v))

        frames.append(
            LCDFrame(
                t=t,
                m=m,
                n_true=n_true,
                R_sorted=R_sorted,
                W=W,
                r_t=r_t,
                l_t=l_t,
                receiver_k=k,
                edges_true=edges_true.copy(),
            )
        )

    # for terminal printing: (t -> k)
    pair_list = [(t, frames[t - 1].receiver_k) for t in range(1, N + 1)]
    return frames, pair_list


def animate_lcd(frames: List[LCDFrame], pair_list, m: int, n_true: int, seed: int, fps: int, print_all: bool):
    interval = int(1000 / fps)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.5], height_ratios=[1.0, 1.0])

    ax_chord = fig.add_subplot(gs[0, 0])   # [0,1] chord view
    ax_sorted = fig.add_subplot(gs[1, 0])  # sorted right endpoints and W boundaries
    ax_deg = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[1, 1])

    pos = np.random.default_rng(seed).normal(scale=0.7, size=(n_true + 1, 2))
    printed_until = 1
    N = len(frames)

    def print_pairs_up_to(t: int):
        nonlocal printed_until
        if print_all:
            print(f"\n=== LCD attach list up to t={t} (rank t -> receiver rank k) ===")
            for s in range(1, t + 1):
                k = pair_list[s - 1][1]
                print(
                    f"t={s:3d}:  R_rank={s:3d} -> k={k:3d}"
                    f" | true v({s})={true_vertex_of(s,m)} -> v({k})={true_vertex_of(k,m)}"
                )
            print("=== end ===\n", flush=True)
            printed_until = t + 1
        else:
            while printed_until <= t:
                s = printed_until
                k = pair_list[s - 1][1]
                print(
                    f"t={s:3d}:  R_rank={s:3d} -> k={k:3d}"
                    f" | true v({s})={true_vertex_of(s,m)} -> v({k})={true_vertex_of(k,m)}",
                    flush=True
                )
                printed_until += 1

    def draw_chord_panel(ax, fr: LCDFrame):
        ax.clear()
        ax.set_title("LCD: chord endpoints on [0,1] (current chord highlighted)", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("x in [0,1]")

        # Place l at y=0.25, r at y=0.75
        ax.scatter([fr.l_t], [0.25], s=80, marker="o", c=[color_for_int(true_vertex_of(fr.receiver_k, fr.m))],
                   edgecolors="k", linewidths=0.5, zorder=3, label="L (current)")
        ax.scatter([fr.r_t], [0.75], s=90, marker="s", c=[color_for_int(true_vertex_of(fr.t, fr.m))],
                   edgecolors="k", linewidths=0.5, zorder=3, label="R (current)")

        ax.plot([fr.l_t, fr.r_t], [0.25, 0.75], lw=2.5, alpha=0.9)

        ax.axvline(fr.l_t, lw=1.0, alpha=0.25)
        ax.axvline(fr.r_t, lw=1.0, alpha=0.25)

        ax.text(
            0.02, 0.95,
            f"rank t={fr.t}/{N}\n"
            f"r_t={fr.r_t:.4f}, l_t={fr.l_t:.4f}\n"
            f"receiver rank k={fr.receiver_k}\n"
            f"true edge: v(t)->v(k) = {true_vertex_of(fr.t, fr.m)}->{true_vertex_of(fr.receiver_k, fr.m)}",
            transform=ax.transAxes,
            fontsize=9,
            va="top"
        )

    def draw_sorted_panel(ax, fr: LCDFrame):
        ax.clear()
        ax.set_title("Sorted right endpoints R_1..R_N with W_i boundaries", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("x in [0,1]")

        # Draw all W boundaries (every mth right endpoint)
        for Wi in fr.W:
            ax.axvline(Wi, lw=0.8, alpha=0.12)

        # highlight current right endpoint location (R_t = r_t)
        ax.axvline(fr.r_t, lw=2.0, alpha=0.75)
        # highlight receiver boundary R_k
        Rk = fr.R_sorted[fr.receiver_k - 1]
        ax.axvline(Rk, lw=2.0, alpha=0.35, linestyle="--")

        # a small dot strip of R's (subsample if too many)
        Nloc = len(fr.R_sorted)
        step = max(1, Nloc // 200)
        Rs = fr.R_sorted[::step]
        ax.scatter(Rs, [0.5] * len(Rs), s=10, alpha=0.35)

        ax.text(
            0.02, 0.92,
            f"W_i = R_(m*i)  (m={fr.m})\n"
            f"solid line: R_t (current)\n"
            f"dashed line: R_k (receiver bound)\n"
            f"k <= t usually (because l_t <= r_t)",
            transform=ax.transAxes,
            fontsize=9,
            va="top"
        )

    def draw_degrees_panel(ax, fr: LCDFrame):
        ax.clear()
        ax.set_title("Degree distribution on TRUE vertices (LCD-induced edges)", fontsize=11)
        ax.set_xlabel("true vertex id (1..n)")
        ax.set_ylabel("degree")

        deg = degrees_from_edges(n_true, fr.edges_true)
        xs = np.arange(1, n_true + 1)
        cols = [color_for_int(i) for i in xs]
        bars = ax.bar(xs, [deg[i] for i in xs], color=cols, edgecolor="none")

        active_max = min(n_true, true_vertex_of(fr.t, m))
        for i in range(active_max + 1, n_true + 1):
            bars[i - 1].set_color((0.85, 0.85, 0.85))

        ax.set_ylim(0, max(1, max(deg) + 2))

    def draw_graph_panel(ax, fr: LCDFrame):
        nonlocal pos
        ax.clear()
        ax.set_title("True-vertex graph (spring layout)", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        active_max = min(n_true, true_vertex_of(fr.t, m))

        edges0 = [(a - 1, b - 1) for (a, b) in fr.edges_true if 1 <= a <= n_true and 1 <= b <= n_true]
        pos0 = pos[1:].copy()
        pos0 = spring_layout_incremental(n=n_true, edges=edges0, pos=pos0, iters=7, seed=seed, step=0.02)
        pos[1:] = pos0

        for a, b in fr.edges_true:
            if a > active_max or b > active_max:
                continue
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.plot([xa, xb], [ya, yb], lw=1.0, alpha=0.45)

        xs = [pos[i, 0] for i in range(1, active_max + 1)]
        ys = [pos[i, 1] for i in range(1, active_max + 1)]
        cols = [color_for_int(i) for i in range(1, active_max + 1)]
        ax.scatter(xs, ys, s=140, c=cols, edgecolors="k", linewidths=0.4, zorder=3)

        for i in range(1, active_max + 1):
            ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=9, ha="center", va="center", zorder=4)

        # highlight newest edge
        if fr.edges_true:
            a, b = fr.edges_true[-1]
            if a <= active_max and b <= active_max:
                ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], lw=3.0, alpha=0.9)

    def update(k):
        fr = frames[k]
        t = fr.t

        print_pairs_up_to(t)

        draw_chord_panel(ax_chord, fr)
        draw_sorted_panel(ax_sorted, fr)
        draw_degrees_panel(ax_deg, fr)
        draw_graph_panel(ax_graph, fr)

        fig.suptitle(f"[LCD] t={t}/{N}  (terminal prints: rank t -> receiver rank k)", fontsize=14)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["half", "lcd", "standard"], required=True)

    parser.add_argument("--n", type=int, default=30, help="number of TRUE vertices after contraction")
    parser.add_argument("--m", type=int, default=3, help="contraction size m (N=mn chords)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    # half-specific (display only)
    parser.add_argument("--pool_show", type=int, default=260, help="HALF only: show last pool_show endpoints in pool panel")

    # printing control
    parser.add_argument("--print_all", action="store_true",
                        help="print entire list up to current t every frame (very verbose)")

    args = parser.parse_args()

    if args.mode == "half":
        frames, pair_list = simulate_half_model(args.m, args.n, args.seed, pool_show=args.pool_show)
        animate_half(frames, pair_list, args.m, args.n, args.seed, args.fps, args.print_all)

    elif args.mode == "lcd":
        frames, pair_list = simulate_lcd_model(args.m, args.n, args.seed)
        animate_lcd(frames, pair_list, args.m, args.n, args.seed, args.fps, args.print_all)

    elif args.mode == "standard":
        frames, pair_list = simulate_standard_ba_model(args.m, args.n, args.seed)
        animate_standard(frames, pair_list, args.m, args.n, args.seed, args.fps, args.print_all)


if __name__ == "__main__":
    main()




#python Ba.py --mode half --n 40 --m 3 --fps 30 --seed 0 --pool_show 300
#python Ba.py --mode lcd --n 40 --m 3 --fps 30 --seed 0
#python Ba.py --mode standard --n 40 --m 3 --fps 30 --seed 0
